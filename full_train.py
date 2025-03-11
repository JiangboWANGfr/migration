import os
import sys
import subprocess
import argparse
import time
import platform
from pathlib import Path
from lib.config import cfg


def submit_job(slurm_args):
    """Submit a job using sbatch and return the job ID."""
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]
    print(f"submitted job {job_id}")

    return job_id


def is_job_finished(job_id):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job_id, '--format=State',
                            '--noheader', '--parsable2'], capture_output=True, text=True)
    # Get job state
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""


def setup_dirs(images, depths, masks, colmap, chunks, output, project):
    images_dir = "../rectified/images" if images == "" else images
    depths_dir = "../rectified/depths" if depths == "" else depths
    if masks == "":
        if os.path.exists(os.path.join(project, "camera_calibration/rectified/masks")):
            masks_dir = "../rectified/masks"
        else:
            masks_dir = ""
    else:
        masks_dir = masks
    colmap_dir = os.path.join(
        project, "camera_calibration", "aligned") if colmap == "" else colmap
    chunks_dir = os.path.join(
        project, "camera_calibration", "chunks") if chunks == "" else chunks
    output_dir = os.path.join(project, "output") if output == "" else output

    return images_dir, depths_dir, masks_dir, colmap_dir, chunks_dir, output_dir


if __name__ == '__main__':
    os_name = platform.system()
    f_path = Path(__file__)
    print(f"Running on {os_name} at {f_path.parent}")
    start_time = time.time()

    if not os.path.exists(cfg.output_dir):
        print(f"creating output dir: {cfg.output_dir}")
        os.makedirs(os.path.join(cfg.output_dir, "scaffold"))
        os.makedirs(os.path.join(cfg.output_dir, "trained_chunks"))

    slurm_args = ["sbatch"]
    # First step is coarse optimization to generate a scaffold that will be used later.
    if cfg.skip_if_exists and os.path.exists(os.path.join(cfg.output_dir, "scaffold/point_cloud/iteration_30000/point_cloud.ply")):
        print("Skipping coarse")
    else:
        if cfg.use_slurm:
            if cfg.extra_training_args != "":
                print(
                    "\nThe script does not support passing extra_training_args to slurm!!\n")
            submitted_jobs_ids = []

            coarse_train = submit_job(slurm_args + [
                f"--error={cfg.output_dir}/scaffold/log.err", f"--output={cfg.output_dir}/scaffold/log.out",
                "coarse_train.slurm", cfg.env_name, cfg.colmap_dir, cfg.images_dir, cfg.output_dir, " --alpha_masks " + cfg.masks_dir
            ])
            print("waiting for coarse training to finish...")
            while is_job_finished(coarse_train) == "":
                time.sleep(10)
        else:
            train_coarse_args = " ".join([
                "python", "train_coarse.py",
                "--cfg_file", "./configs/train_coarse.yaml"
            ])
            try:
                subprocess.run(train_coarse_args, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing train_coarse: {e}")
                sys.exit(1)

    if not os.path.isabs(cfg.images_dir):
        images_dir = os.path.join("../", cfg.images_dir)
    if not os.path.isabs(cfg.depths_dir):
        depths_dir = os.path.join("../", cfg.depths_dir)
    if cfg.masks_dir != "" and not os.path.isabs(cfg.masks_dir):
        masks_dir = os.path.join("../", cfg.masks_dir)

    # Now we can train each chunks using the scaffold previously created
    train_chunk_args = " ".join([
        "python", "-u train_single.py",
        "--cfg_file", "./configs/train_coarse.yaml"
    ])
    # if masks_dir != "":
    #     train_chunk_args += " --alpha_masks " + masks_dir
    # if cfg.extra_training_args != "":
    #     train_chunk_args += " " + cfg.extra_training_args

    hierarchy_creator_args = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyCreator.exe " if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyCreator "
    hierarchy_creator_args = os.path.join(
        f_path.parent.parent, hierarchy_creator_args)

    post_opt_chunk_args = " ".join([
        "python", "-u train_post.py",
        "--cfg_file", "./configs/train_coarse.yaml"
    ])

    chunk_names = os.listdir(cfg.chunks_dir)
    for chunk_name in chunk_names:
        source_chunk = os.path.join(cfg.chunks_dir, chunk_name)
        trained_chunk = os.path.join(
            cfg.output_dir, "trained_chunks", chunk_name)

        if cfg.skip_if_exists and os.path.exists(os.path.join(trained_chunk, "hierarchy.hier_opt")):
            print(f"Skipping {chunk_name}")
        else:
            # Training can be done in parallel using slurm.
            if cfg.use_slurm:
                job_id = submit_job(slurm_args + [
                    f"--error={trained_chunk}/log.err", f"--output={trained_chunk}/log.out",
                    "train_chunk.slurm", source_chunk, cfg.output_dir, cfg.env_name,
                    chunk_name, hierarchy_creator_args, images_dir,
                    depths_dir, " --alpha_masks " + masks_dir
                ])

                submitted_jobs_ids.append(job_id)
            else:
                print(f"Training chunk {chunk_name}")
                try:
                    subprocess.run(
                        train_chunk_args,
                        shell=True, check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error executing train_single: {e}")
                    if not cfg.keep_running:
                        sys.exit(1)

                # Generate a hierarchy within each chunks
            print(f"Generating hierarchy for chunk {chunk_name}")
            try:
                subprocess.run(
                    hierarchy_creator_args + " ".join([
                        os.path.join(
                            trained_chunk, "point_cloud/iteration_30000/point_cloud.ply"),
                        source_chunk,
                        trained_chunk,
                        os.path.join(cfg.output_dir,
                                     "scaffold/point_cloud/iteration_30000")
                    ]),
                    shell=True, check=True, text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing hierarchy_creator: {e}")
                if not cfg.keep_running:
                    sys.exit(1)

            # Post optimization on each chunks
            print(f"post optimizing chunk {chunk_name}")
            try:
                subprocess.run(
                    post_opt_chunk_args + " -s " + source_chunk +
                    " --model_path " + trained_chunk +
                    " --hierarchy " +
                    os.path.join(trained_chunk, "hierarchy.hier"),
                    shell=True, check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing train_post: {e}")
                if not cfg.keep_running:
                    # TODO: log where it fails and don't add it to the consolidation and add a warning at the end
                    sys.exit(1)

    if cfg.use_slurm:
        # Check every 10 sec all the jobs status
        all_finished = False
        all_status = []
        last_count = 0
        print(f"Waiting for chunks to be trained in parallel ...")

        while not all_finished:
            # print("Checking status of all jobs...")
            all_status = [is_job_finished(
                id) for id in submitted_jobs_ids if is_job_finished(id) != ""]
            if last_count != all_status.count("COMPLETED"):
                last_count = all_status.count("COMPLETED")
                print(f"processed [{last_count} / {len(chunk_names)} chunks].")

            all_finished = len(all_status) == len(submitted_jobs_ids)

            if not all_finished:
                time.sleep(10)  # Wait before checking again

        if not all(status == "COMPLETED" for status in all_status):
            print("At least one job failed or was cancelled, check at error logs.")

    end_time = time.time()
    print(f"Successfully trained in {(end_time - start_time)/60.0} minutes.")

    # Consolidation to create final hierarchy
    hierarchy_merger_path = "submodules/gaussianhierarchy/build/Release/GaussianHierarchyMerger.exe" if os_name == "Windows" else "submodules/gaussianhierarchy/build/GaussianHierarchyMerger"
    hierarchy_merger_path = os.path.join(
        f_path.parent.parent, hierarchy_merger_path)

    consolidation_args = [
        hierarchy_merger_path, f"{cfg.output_dir}/trained_chunks",
        "0", cfg.chunks_dir, f"{cfg.output_dir}/merged.hier"
    ]

    consolidation_args = consolidation_args + chunk_names
    print(f"Consolidation...")
    if cfg.use_slurm:
        consolidation = submit_job(slurm_args + [
            f"--error={cfg.output_dir}/consolidation_log.err", f"--output={cfg.output_dir}/consolidation_log.out",
            "consolidate.slurm"] + consolidation_args)

        while is_job_finished(consolidation) == "":
            time.sleep(10)
    else:
        try:
            subprocess.run(consolidation_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing consolidation: {e}")
            sys.exit(1)

    end_time = time.time()
    print(
        f"Total time elapsed for training and consolidation {(end_time - start_time)/60.0} minutes.")
