import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest


@pytest.mark.smoke
@pytest.mark.slow
def test_ebproc():

    with TemporaryDirectory() as save_dir:

        input_dir = Path(__file__).parent / "input"

        # Run the ebproc command
        result = subprocess.run(
            [
                "ebproc",
                "--data-direc",
                input_dir,
                "--save-direc",
                str(save_dir),
                "--save-figs",
                "--land",
                input_dir / "reproj_land.tiff",
            ],
            capture_output=True,
            text=True,
        )

        # Check command ran successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"

        # Check output files were created
        for folder in ["214", "215"]:
            assert any(
                Path(save_dir, folder).iterdir()
            ), f"No files were created in the output directory {folder}."
