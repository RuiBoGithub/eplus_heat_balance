
import os
import subprocess

def run_IDF(idf_file, weather_file, energyplus_exe="C:/EnergyPlusV23-2-0/EnergyPlus"):
    """
    Runs an EnergyPlus simulation and saves results to an output folder.

    Parameters:
        idf_file (str): Path to the IDF file.
        weather_file (str): Path to the weather file (.epw).
        energyplus_exe (str): Path to the EnergyPlus executable.

    Returns:
        str: Path to the output folder where results are stored.
    """
    # Setup paths
    base_path = os.path.dirname(idf_file)
    idf_name = os.path.splitext(os.path.basename(idf_file))[0]
    output_folder = os.path.join(base_path, idf_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Run EnergyPlus simulation
    command = [
        energyplus_exe,
        "--readvars",
        "--output-directory", output_folder,
        "--weather", weather_file,
        idf_file
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error running simulation:", e.output)
        return None

    # Convert the path format for output
    single_slash_path = output_folder.replace("\\", "\\")
    print("Output Folder:", single_slash_path)

    return output_folder
