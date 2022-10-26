from bing_images import bing

bing.download_images("walleye",
                      20,
                      output_dir="D:\CP\Projects\FishApp\FishDatabase\Images",
                      pool_size=10,
                      file_type="png",
                      force_replace=True)