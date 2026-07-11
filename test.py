import logging
from geecs_data_utils.config_roots import image_analysis_config

logging.basicConfig(level=logging.INFO)

# Swap out the broken ScanPaths calls with your actual local folder paths
# local_paths_config = r"C:\Users\loasis.LOASIS\Desktop\Scripts\People\Rachel\git_repos\GEECS-Plugins\path_to_your_configs"
# local_image_configs = r"C:\Users\loasis.LOASIS\Desktop\Scripts\People\Rachel\git_repos\GEECS-Plugins\path_to_image_analysis_configs"

# image_analysis_config.set_base_dir(local_paths_config, local_image_configs)

import logging

from geecs_data_utils.scan_data import ScanPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("image_analysis").setLevel(logging.WARNING)
logging.getLogger("geecs_data_utils").setLevel(logging.WARNING)

image_analysis_config.set_base_dir(ScanPaths.paths_config.image_analysis_configs_path)
