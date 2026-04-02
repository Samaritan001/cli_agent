import docker
import os
import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("docker_image_builder")

def build_and_save_image(image_name, tag, source_path, save_path):
    client = docker.from_env()
    
    logger.info(f"--- Starting build for {image_name}:{tag} ---")
    
    try:
        # 1. Build the image
        # 'path' is the directory containing the Dockerfile
        logger.warning("starting")
        image, build_logs = client.images.build(
            path=source_path,
            dockerfile="src/cli_server/weather_py_server/Dockerfile",
            tag=f"{image_name}:{tag}",
            rm=True  # Remove intermediate containers
        )
        logger.warning("image built")
        
        # Optional: logger.info build logs
        for line in build_logs:
            if 'stream' in line:
                logger.info(line['stream'].strip())

        logger.info(f"Successfully built {image.tags}")

        # 2. Save the image to a local folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        file_name = f"{image_name}_{tag}.tar"
        full_path = os.path.join(save_path, file_name)
        
        logger.info(f"Saving image to {full_path}...")
        with open(full_path, 'wb') as f:
            for chunk in image.save():
                f.write(chunk)
                
        logger.info("Image saved successfully.")

    except Exception as e:
        logger.info(f"An error occurred: {e}")

if __name__ == "__main__":
    build_and_save_image(
        image_name="weather-server",
        tag="test",
        source_path="../..",
        save_path="./weather_py_server"
    )