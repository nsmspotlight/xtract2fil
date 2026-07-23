import re
import logging
from pathlib import Path
from typing import Generator
from datetime import datetime
from rich.console import Console
from contextlib import ExitStack

import pytz
import cyclopts
import numpy as np
import pandas as pd
import astropy.units as u
from priwo import writehdr
from astropy.time import Time
from joblib import Parallel, delayed
from rich.logging import RichHandler
from astropy.coordinates import SkyCoord, TETE

app = cyclopts.App()
app["--help"].group = "Admin"
app["--version"].group = "Admin"

console = Console()

logging.basicConfig(
    level="INFO",
    datefmt="[%X]",
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
log = logging.getLogger("xtract2fil")


class XtractionError(Exception):
    pass


def getmjd(t: datetime) -> float:
    localtz = pytz.timezone("Asia/Kolkata")
    localdt = localtz.localize(t, is_dst=None)
    utcdt = localdt.astimezone(pytz.utc)
    mjd = Time(utcdt).mjd
    mjd = np.asarray(mjd)
    mjd = float(mjd)
    return mjd


def inchunks(fx, N: int) -> Generator:
    while True:
        data = fx.read(N)
        if (not data) or len(data) < N:
            break
        yield data


def asciihdr(fn: str | Path) -> dict:
    hdr = {}
    extras = []
    with open(fn, "r") as lines:
        for line in lines:
            if line.startswith(("#", " ")):
                extras.append(line)
                continue
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()
            try:
                # fmt: off
                name, conv = {
                    "Header file": ("fname", str),
                    "Beam ID": ("beamid", int),
                    "Host ID": ("hostid", int),
                    "Host name": ("hostname", str),
                    "GTAC code": ("gtaccode", str),
                    "Observer": ("observer", str),
                    "GTAC title": ("gtactitle", str),
                    "Source": ("source", str),
                    "Source RA (Rad)": ("ra", float),
                    "Source DEC (Rad)": ("dec", float),
                    "Channels": ("nf", int),
                    "Bandwidth (MHz)": ("bw", float),
                    "Channel width (Hz)": ("df", lambda x: float(x) * 1e-6),
                    "Frequency Ch. 0  (Hz)": ("f0", lambda x: float(x) * 1e-6),
                    "Sampling time  (uSec)": ("dt", lambda x: float(x) * 1e-6),
                    "Antennas pol1": ("antX", lambda x: list(x.split())),
                    "Antennas pol2": ("antY", lambda x: list(x.split())),
                    "Antenna mask pol1": ("maskX", lambda x: [int(_) == 1 for _ in np.binary_repr(int(x, 0))]),
                    "Antenna mask pol2": ("maskY", lambda x: [int(_) == 1 for _ in np.binary_repr(int(x, 0))]),
                    "Beam mode": ("beammode", str),
                    "No. of stokes": ("npol", int),
                    "Num bits/sample": ("nbits", int),
                    "Total No. of Beams": ("nbeams", int),
                    "No. of PC Baselines": ("nbaselines", int),
                    "Total No. of Beams/host": ("nbeamsperhost", int),
                    "De-Disperion DM": ("dm", lambda x: None if x == "NA" else float(x)),
                    "Date": ("istdate", str),
                    "IST Time": ("isttime", str),
                }[key]
                # fmt: on
                hdr[name] = conv(val)
            except KeyError:
                pass

    if hdr.get("istdate", None) is None:
        raise XtractionError("DATE FIELD MISSING. FILE CORRUPTED. ABORT.")
    if hdr.get("isttime", None) is None:
        raise XtractionError("IST TIME FIELD MISSING. FILE CORRUPTED. ABORT.")
    if hdr.get("nbeams", None) is None:
        raise XtractionError("NBEAMS FIELD MISSING. FILE CORRUPTED. ABORT.")

    fields = []
    header = extras[1].split()[2:]
    extras = extras[2:]
    for extra in extras:
        fields.append([float(_) for _ in extra.split()])
    hdr["coords"] = pd.DataFrame(fields, columns=header)
    ist = " ".join([hdr["istdate"], hdr["isttime"][:-3]])
    hdr["istdatetime"] = datetime.strptime(ist, "%d/%m/%Y %H:%M:%S.%f")
    return hdr


def ra2flt(coords: SkyCoord) -> float:
    ra_d, ra_m, ra_s = getattr(getattr(coords, "ra"), "hms")
    ra_d = int(ra_d)
    ra_m = int(ra_m)
    ra_s = float(ra_s)
    if ra_d < 0.0:
        ra_m = -ra_m
        ra_s = -ra_s
    ra_d = str(ra_d).zfill(2)
    ra_m = str(ra_m).zfill(2)
    ra_si, ra_sf = str(ra_s).split(".")
    ra_s = ".".join([ra_si.zfill(2), ra_sf])
    ra_f = float("".join([ra_d, ra_m, ra_s]))
    return ra_f


def dec2flt(coords: SkyCoord) -> float:
    dec_d, dec_m, dec_s = getattr(getattr(coords, "dec"), "dms")
    dec_d = int(dec_d)
    dec_m = int(dec_m)
    dec_s = float(dec_s)
    if dec_m < 0.0:
        dec_m = -dec_m
    if dec_s < 0.0:
        dec_s = -dec_s
    dec_d = str(dec_d).zfill(2)
    dec_m = str(dec_m).zfill(2)
    dec_si, dec_sf = str(dec_s).split(".")
    dec_s = ".".join([dec_si.zfill(2), dec_sf])
    dec_f = float("".join([dec_d, dec_m, dec_s]))
    return dec_f


def iaxtract(fn: Path) -> None:
    hdr = asciihdr(fn.with_suffix(".0.ahdr"))
    data = np.fromfile(fn, dtype=np.uint8)
    data = data.reshape(-1, hdr["nf"])

    nf = hdr["nf"]
    fh = hdr["f0"]
    df = hdr["df"]
    dt = hdr["dt"]
    nbits = hdr["nbits"]
    fname = str(fn.name)
    source = hdr["source"]
    mjd = getmjd(hdr["istdatetime"])

    rad = getattr(u, "rad")
    obstime = Time(mjd, format="mjd")
    coords = SkyCoord(
        hdr["ra"] * rad,
        hdr["dec"] * rad,
        frame=TETE(obstime=obstime),
    ).transform_to("icrs")
    ra_f = ra2flt(coords)
    dec_f = dec2flt(coords)

    nblks = 32
    defaultdt = 1.31072e-3
    blktime = 800 * defaultdt
    nt = int(round(blktime / dt))
    slicesize = nf * nt * nblks

    fbin = int(nf / 1024)
    tbin = int(13.1072e-3 / dt)
    if (nf % fbin) != 0:
        raise ValueError(f"fbin={fbin} must be a factor of nf={nf}")
    elif (nt % tbin) != 0:
        raise ValueError(f"tbin={tbin} must be a factor of nt={nt}")

    filhdr = {
        "rawdatafile": fname,
        "source_name": source,
        "az_start": 0.0,
        "za_start": 0.0,
        "src_raj": ra_f,
        "src_dej": dec_f,
        "tstart": mjd,
        "tsamp": dt,
        "fch1": fh,
        "foff": df,
        "nchans": nf,
        "telescope_id": 7,
        "machine_id": 14,
        "data_type": 1,
        "ibeam": 1,
        "nbeams": 1,
        "nbits": nbits,
        "barycentric": 0,
        "pulsarcentric": 0,
        "nifs": 1,
        "size": 0,
    }

    writehdr(filhdr, fn.with_suffix(".fil"))

    filhdr["foff"] = df * fbin
    filhdr["tsamp"] = dt * tbin
    filhdr["nchans"] = int(round(nf / fbin))
    writehdr(filhdr, fn.with_suffix(".down.fil"))

    with ExitStack() as stack:
        filfile = stack.enter_context(open(fn.with_suffix(".fil"), "ab"))
        dwnfile = stack.enter_context(open(fn.with_suffix(".down.fil"), "ab"))
        with open(fn, "rb") as f:
            for data in inchunks(f, slicesize):
                array = np.frombuffer(data, dtype=np.uint8)
                array = array.reshape(-1, nf)
                array.tofile(filfile)

                array = array.reshape((-1, int(array.shape[1] // fbin), fbin)).mean(2)
                array = array.reshape((-1, tbin, array.shape[1])).mean(1)
                array = array.astype(np.uint8)
                array.tofile(dwnfile)


def pcxtract(fn: Path, fildir: Path, dwndir: Path):
    hdr = asciihdr(str(fn) + ".ahdr")

    nbeams = hdr["nbeamsperhost"]

    nf = hdr["nf"]
    fh = hdr["f0"]
    df = hdr["df"]
    dt = hdr["dt"]
    nbits = hdr["nbits"]
    fname = str(fn.name)
    source = hdr["source"]
    radecs = hdr["coords"]
    mjd = getmjd(hdr["istdatetime"])

    nblks = 32
    defaultdt = 1.31072e-3
    blktime = 800 * defaultdt
    nt = int(round(blktime / dt))
    slicesize = nf * nt * nblks

    fbin = int(nf / 1024)
    tbin = int(13.1072e-3 / dt)
    if (nf % fbin) != 0:
        raise ValueError(f"fbin={fbin} must be a factor of nf={nf}")
    elif (nt % tbin) != 0:
        raise ValueError(f"tbin={tbin} must be a factor of nt={nt}")

    rad = getattr(u, "rad")
    beamix = radecs["BM-Idx"].to_numpy(dtype=int)
    filpaths = [fildir / f"BM{ix}.fil" for ix in beamix]
    dwnpaths = [dwndir / f"BM{ix}.down.fil" for ix in beamix]
    for ix, filpath in enumerate(filpaths):
        obstime = Time(mjd, format="mjd")
        coords = SkyCoord(
            radecs.iloc[ix]["RA"] * rad,
            radecs.iloc[ix]["DEC"] * rad,
            frame=TETE(obstime=obstime),
        ).transform_to("icrs")
        ra_f = ra2flt(coords)
        dec_f = dec2flt(coords)
        filhdr = {
            "rawdatafile": fname,
            "source_name": source,
            "nifs": 1,
            "nbits": nbits,
            "data_type": 1,
            "machine_id": 7,
            "telescope_id": 7,
            "barycentric": 0,
            "pulsarcentric": 0,
            "tstart": mjd,
            "foff": df,
            "fch1": fh,
            "tsamp": dt,
            "nchans": nf,
            "src_raj": ra_f,
            "src_dej": dec_f,
            "size": 0,
        }
        writehdr(filhdr, str(filpath))

        filhdr["foff"] = df * fbin
        filhdr["tsamp"] = dt * tbin
        filhdr["nchans"] = int(round(nf / fbin))
        writehdr(filhdr, str(dwnpaths[ix]))

    with ExitStack() as stack:
        filfiles = [stack.enter_context(open(_, "ab")) for _ in filpaths]
        dwnfiles = [stack.enter_context(open(_, "ab")) for _ in dwnpaths]
        with open(fn, "rb") as f:
            for ix, data in enumerate(inchunks(f, slicesize)):
                array = np.frombuffer(data, dtype=np.uint8)
                array = array.reshape(-1, nf)
                array.tofile(filfiles[ix % nbeams])

                array = array.reshape((-1, int(array.shape[1] // fbin), fbin)).mean(2)
                array = array.reshape((-1, tbin, array.shape[1])).mean(1)
                array = array.astype(np.uint8)
                array.tofile(dwnfiles[ix % nbeams])


@app.command
def ia(obsname: str):
    obsdir = Path("/lustre_data/spotlight/data") / obsname
    scans = set([_.name.split(".")[0] for _ in (obsdir / "IABeamData").glob("*.raw*")])

    rxraw = re.compile(r"\.raw\.\d+$")
    rxhdr = re.compile(r"\.raw\.\d+\.ahdr$")

    rawfiles = [
        f
        for scan in scans
        if (f := obsdir / "IABeamData" / f"{scan}.raw.*").is_file()
        and rxraw.search(f.name)
    ]

    hdrfiles = [
        f
        for scan in scans
        if (f := obsdir / "IABeamData" / f"{scan}.raw.*.ahdr").is_file()
        and rxhdr.search(f.name)
    ]

    if (len(rawfiles) != len(scans)) or (len(hdrfiles) != len(scans)):
        raise XtractionError("MISSING FILES. ABORT.")
    if len(rawfiles) != len(hdrfiles):
        raise XtractionError("NUMBER OF RAW AND HDR FILES NOT EQUAL. ABORT.")

    njobs = len(scans)
    log.info(f"Xtracting IA beam data for {obsdir}...")
    log.info(f"Number of rawfiles = {len(rawfiles):d}.")
    log.info(f"Number of cores being used = {njobs:d} cores.")
    Parallel(n_jobs=njobs)(delayed(iaxtract)(fn=fn) for fn in rawfiles)


@app.command
def pc(obsname: str):
    nhosts = 16
    obsdir = Path("/lustre_data/spotlight/data") / obsname
    scans = set([_.name.split(".")[0] for _ in (obsdir / "BeamData").glob("*.raw*")])

    rxraw = re.compile(r"\.raw\.\d+$")
    rxhdr = re.compile(r"\.raw\.\d+\.ahdr$")
    log.info(f"Xtracting PC beam data for {obsdir}...")
    for scan in scans:
        rawfiles = [
            f
            for f in (obsdir / "BeamData").glob(f"{scan}.raw.*")
            if f.is_file() and rxraw.search(f.name)
        ]
        hdrfiles = [
            f
            for f in (obsdir / "BeamData").glob(f"{scan}.raw.*.ahdr")
            if f.is_file() and rxhdr.search(f.name)
        ]
        if (len(rawfiles) != nhosts) or (len(hdrfiles) != nhosts):
            raise XtractionError("NOT ENOUGH FILES. ABORT.")
        if len(rawfiles) != len(hdrfiles):
            raise XtractionError("NUMBER OF RAW AND HDR FILES NOT EQUAL. ABORT.")

        fildir = Path(obsdir) / "FilData" / scan
        dwndir = Path(obsdir) / "FilData_dwnsmp" / scan
        fildir.mkdir(exist_ok=True)
        dwndir.mkdir(exist_ok=True)

        njobs = len(rawfiles)
        log.info(f"Xtracting data for {scan}...")
        log.info(f"Number of rawfiles = {len(rawfiles):d}.")
        log.info(f"Number of cores being used = {njobs:d} cores.")
        Parallel(n_jobs=njobs)(
            delayed(pcxtract)(
                fn=fn,
                fildir=fildir,
                dwndir=dwndir,
            )
            for fn in rawfiles
        )


if __name__ == "__main__":
    app()
