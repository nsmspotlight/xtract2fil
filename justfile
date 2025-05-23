pkg  := "xtract2fil"
desc := "eXtract raw data from the SPOTLIGHT system to SIGPROC filterbank files."

alias l := loc
alias t := test
alias d := docs
alias c := clean
alias i := install
alias u := uninstall


# List available commands.
default:
    #!/usr/bin/env python
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console

    console = Console()

    grid = Table.grid(expand=True, padding=(0, 2, 0, 2))
    grid.add_column(justify="left", style="bold")
    grid.add_column(justify="right", style="italic")

    grid.add_row("loc ([i]l[/i])", "Chart LOCs")
    grid.add_row("clean ([i]c[/i])", "Clean up")
    grid.add_row("test ([i]t[/i])", "Run tests")
    grid.add_row("install ([i]i[/i])", "Install")
    grid.add_row("uninstall ([i]u[/i])", "Uninstall")
    grid.add_row("docs ([i]d[/i])", "Build docs.")

    console.print(
        Panel(
            grid,
            padding=2,
            expand=False,
            title="[b]{{pkg}}[/b]: [i]{{desc}}[/i]",
        )
    )

# Chart LOCs.
@loc:
    echo "Charting LOCs..."
    tokei src tests -o json | tokei-pie

# Clean up.
@clean:
    echo "Cleaning..."
    rm -rf tmp
    rm -rf dist
    rm -rf build
    rm -rf .eggs
    rm -rf .coverage
    rm -rf .mypy_cache
    rm -rf docs/build/*
    rm -rf .pytest_cache
    fd -I -e pyc -x rm -rf
    fd -I __pycache__ -x rm -rf

# Run tests.
@test: && clean
	ward

# Install.
@install: && clean
    echo "Installing..."
    pip install -e .

# Uninstall.
@uninstall: && clean
    echo "Uninstalling {{pkg}}..."
    pip uninstall {{pkg}}
    rm -rf src/{{pkg}}.egg-info
    rm -rf src/{{pkg}}/_version.py

# Build docs.
@docs:
    echo "Building docs for {{pkg}}..."
    sphinx-reload docs
