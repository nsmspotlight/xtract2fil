import logging
from pathlib import Path
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
from astropy.coordinates import SkyCoord

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


def read_asciihdr(fn: str | Path) -> dict:
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
                    "Frequency Ch. 0  (Hz)": ("f0", lambda x: float(x) * 1e-6),
                    "Channel width (Hz)": ("df", lambda x: float(x) * 1e-6),
                    "Sampling time  (uSec)": ("dt", lambda x: float(x) * 1e-6),
                    "Antenna mask pol1": ("maskX", lambda x: [int(_) == 1 for _ in np.binary_repr(int(x, 0))]),
                    "Antennas pol1": ("antX", lambda x: list(x.split())),
                    "Antenna mask pol2": ("maskY", lambda x: [int(_) == 1 for _ in np.binary_repr(int(x, 0))]),
                    "Antennas pol2": ("antY", lambda x: list(x.split())),
                    "Beam mode": ("beammode", str),
                    "No. of stokes": ("npol", int),
                    "Num bits/sample": ("nbits", int),
                    "De-Disperion DM": ("dm", lambda x: None if x == "NA" else float(x)),
                    "PFB": ("pfb", bool),
                    "PFB number of taps": ("pfbtaps", lambda x: None if x == "NA" else float(x)),
                    "WALSH": ("walsh", lambda x: False if x == "OFF" else True),
                    "Broad-band RFI Filter": ("rfifilter", lambda x: False if x == "OFF" else True),
                    "Date": ("istdate", str),
                    "IST Time": ("isttime", str),
                }[key]
                # fmt: on
                hdr[name] = conv(val)
            except KeyError:
                pass

    fields = []
    header = extras[1].split()[2:]
    extras = extras[2:]
    for extra in extras:
        fields.append([float(_) for _ in extra.split()])
    hdr["coords"] = pd.DataFrame(fields, columns=header)
    ist = " ".join([hdr["istdate"], hdr["isttime"][:-3]])
    hdr["istdatetime"] = datetime.strptime(ist, "%d/%m/%Y %H:%M:%S.%f")
    return hdr


def getmjd(t: datetime):
    localtz = pytz.timezone("Asia/Kolkata")
    localdt = localtz.localize(t, is_dst=None)
    utcdt = localdt.astimezone(pytz.utc)
    mjd = Time(utcdt).mjd
    return mjd


def inchunks(fx, N: int):
    while True:
        data = fx.read(N)
        if (not data) or len(data) < N:
            break
        yield data


def xtract2fil(
    fn: str | Path,
    nbeams: int,
    outdir: str | Path,
    fbin: int = 1,
    tbin: int = 1,
    offset: int = 64,
):
    fn = Path(fn)
    outdir = Path(outdir)

    hdr = read_asciihdr(str(fn) + ".ahdr")

    df = hdr["df"]
    bw = hdr["bw"]
    dt = hdr["dt"]
    nf = hdr["nf"]
    nbits = hdr["nbits"]
    fname = str(fn.name)
    source = hdr["source"]
    radecs = hdr["coords"]
    mjd = getmjd(hdr["istdatetime"])
    flip = True if df > 0.0 else False

    fh = hdr["f0"]
    if flip:
        df = -df
        fh = hdr["f0"] + bw - (0.5 * df)

    nblks = 32
    defaultdt = 1.31072e-3
    blktime = 800 * defaultdt
    nt = int(round(blktime / dt))
    slicesize = nf * nt * nblks
    if (nf % fbin) != 0:
        raise ValueError(f"fbin={fbin} must be a factor of nf={nf}")
    elif (nt % tbin) != 0:
        raise ValueError(f"tbin={tbin} must be a factor of nt={nt}")
    else:
        downflag = (fbin > 1) or (tbin > 1)

    rad = getattr(u, "rad")
    beamix = radecs["BM-Idx"].to_numpy(dtype=int)
    filpaths = [outdir / f"BM{ix}{'.down' if downflag else ''}.fil" for ix in beamix]
    for ix, filpath in enumerate(filpaths):
        coords = SkyCoord(radecs.iloc[ix]["RA"] * rad, radecs.iloc[ix]["DEC"] * rad)

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
        ra_sigproc = float("".join([ra_d, ra_m, ra_s]))

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
        dec_sigproc = float("".join([dec_d, dec_m, dec_s]))

        writehdr(
            {
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
                "foff": df * fbin,
                "fch1": fh,
                "tsamp": dt * tbin,
                "nchans": int(nf / fbin),
                "src_raj": ra_sigproc,
                "src_dej": dec_sigproc,
                "size": 0,
            },
            str(filpath),
        )

    with ExitStack() as stack:
        filfiles = [stack.enter_context(open(_, "ab")) for _ in filpaths]
        with open(fn, "rb") as fx:
            for ix, data in enumerate(inchunks(fx, slicesize)):
                filfile = filfiles[ix % nbeams]
                if offset != 0:
                    array = np.frombuffer(data, dtype=np.int8)
                    array = array.astype(np.int16) + offset
                    array = array.astype(np.uint8)
                else:
                    array = np.frombuffer(data, dtype=np.uint8)
                array = array.reshape(-1, nf)
                if flip:
                    array = np.fliplr(array)
                if downflag:
                    array = array.reshape((-1, tbin, array.shape[1])).mean(1)
                    array = array.reshape((-1, int(array.shape[1] // fbin), fbin)).mean(2)
                    array = array.astype(np.uint8)
                array.tofile(filfile)


@app.default
def main(
    files: list[str | Path],
    /,
    fbin: int = 1,
    tbin: int = 1,
    njobs: int = -1,
    nbeams: int = 10,
    offset: int = 64,
    output: str | Path = Path.cwd(),
):
    njobs = njobs if njobs > 0 else len(files)
    log.info(f"Xtracting {nbeams} beams from {len(files)} files...")
    log.info(f"Using {njobs} cores")
    Parallel(n_jobs=njobs)(
        delayed(xtract2fil)(
            fn=f,
            fbin=fbin,
            tbin=tbin,
            nbeams=nbeams,
            outdir=output,
            offset=offset,
        )
        for f in files
    )


if __name__ == "__main__":
    app()
