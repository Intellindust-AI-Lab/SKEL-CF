import hydra

from vis.mesh_estimator import HumanMeshEstimator



@hydra.main(version_base='1.2', config_path="../config", config_name="vis.yaml")
def main(cfg):

    estimator = HumanMeshEstimator(cfg=cfg)

    estimator.run_on_images(cfg.misc.image_folder, cfg.misc.output_folder)
    
if __name__=='__main__':
    main()
