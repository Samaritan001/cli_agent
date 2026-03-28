import docker
import os
import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("manual_generator")

def build_and_save_image(image_name, tag, source_path, save_path):
    client = docker.from_env()
    
    logging.info(f"--- Starting build for {image_name}:{tag} ---")
    
    try:
        # 1. Build the image
        # 'path' is the directory containing the Dockerfile
        logging.warning("starting")
        image, build_logs = client.images.build(
            path=source_path,
            dockerfile="src/cli_server/weather_py_server/Dockerfile",
            tag=f"{image_name}:{tag}",
            rm=True  # Remove intermediate containers
        )
        logging.warning("image built")
        
        # Optional: logging.info build logs
        for line in build_logs:
            if 'stream' in line:
                logging.info(line['stream'].strip())

        logging.info(f"Successfully built {image.tags}")

        # 2. Save the image to a local folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        file_name = f"{image_name}_{tag}.tar"
        full_path = os.path.join(save_path, file_name)
        
        logging.info(f"Saving image to {full_path}...")
        with open(full_path, 'wb') as f:
            for chunk in image.save():
                f.write(chunk)
                
        logging.info("Image saved successfully.")

    except Exception as e:
        logging.info(f"An error occurred: {e}")

if __name__ == "__main__":
    build_and_save_image(
        image_name="weather-server",
        tag="test",
        source_path="../..",
        save_path="./weather_py_server"
    )