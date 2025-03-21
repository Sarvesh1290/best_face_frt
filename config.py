import subprocess
CONFIG = {
    "source": 0,
    "camera_id": 1
}
 
# Command to run the script
command = (
    f"python Best_face.py --source {CONFIG['source']} "
    f"--camera_id {CONFIG['camera_id'] } "
)

print(f"Executing: {command}")
subprocess.run(command, shell=True)