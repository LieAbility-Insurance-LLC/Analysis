import logging
import subprocess
import importlib

def ensure_packages_installed(packages):
    """
    Ensures required packages are installed in the environment.
    This should be run one time before using the main script.
    """
    for package in packages:
        try:
            importlib.import_module(package)
            logging.info(f"Package '{package}' is already installed.")
        except ImportError:
            logging.warning(f"Package '{package}' not installed. Installing now...")
            subprocess.check_call(["python", "-m", "pip", "install", package])
            logging.info(f"Installed '{package}'. Restart your session if needed.")

if __name__ == "__main__":
    packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "sklearn", "imblearn", "xgboost", "shap"
    ]
    ensure_packages_installed(packages)
