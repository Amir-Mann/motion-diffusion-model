import re
import os
import sys
import torch
import pickle
import argparse
import datetime
import numpy as np
from dataset_info import dataset_torch_std, dataset_torch_mean

sys.path.append('/home/amir.mann/MDM/')
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils.math_utils import perspective_projection_batch, compute_into_camera_shift_from_angle


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained, sample_2d, date_save):
    
    row_file_template = ("2d_" if sample_2d else "") + 'sample{:02d}.mp4'
    all_file_template = ("2d_" if sample_2d else "") + 'samples_{:02d}_to_{:02d}.mp4'
    if date_save:
        date_str = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M')
        row_file_template = date_str + row_file_template
        all_file_template = date_str + all_file_template
    if unconstrained:
        sample_file_template = (date_str if date_save else "") + ("2d_" if sample_2d else "") + 'row{:02d}_col{:02d}.mp4'
        sample_print_template = ("2d_" if sample_2d else "") + '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = ("2d_" if sample_2d else "") + row_file_template.replace('sample', 'row')
        row_print_template = ("2d_" if sample_2d else "") + '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = ("2d_" if sample_2d else "") + all_file_template.replace('samples', 'rows')
        all_print_template = ("2d_" if sample_2d else "") + '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = (date_str if date_save else "") + ("2d_" if sample_2d else "") + 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = ("2d_" if sample_2d else "") + '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = ("2d_" if sample_2d else "") + '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = ("2d_" if sample_2d else "") + '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template

def parse_pickle(data, keys):
    caption = ""
    parsed_data = None
    for key in keys:
        print(key, str(parsed_data)[:10])
        caption += " " + key
        if "eval:" == key[:len("eval:")]:
            key = eval(key[len("eval:"):])
            print("eval inside parse_pickle:", key, type(key))
        if parsed_data is None:
            parsed_data = data[key]
        else:
            parsed_data = parsed_data[key]
    
    #parsed_data = parsed_data * data["distances"][evaled_key]
    if isinstance(parsed_data, torch.Tensor):
        if len(parsed_data.shape) > 3:
            raise RuntimeError(f"Data is of shape {parsed_data.shape} larger then 3, maybe use some index.")
        parsed_data = parsed_data.permute(2, 0 ,1)
        if parsed_data.shape[-1] == 2:
            nframes, njoints, _ = parsed_data.shape
            z_dim = torch.zeros(nframes, njoints, 1, device=parsed_data.device)      
            parsed_data = torch.cat((parsed_data, z_dim), dim=2)
        try: 
            parsed_data = parsed_data.detach().numpy()
        except:
            parsed_data = parsed_data.cpu().numpy()
    return parsed_data, caption

def get_data(args, sample):
    print(sample, args.pickle_keys)
    other_data = None
    if os.path.isfile(sample):
        with open(sample, "rb") as f:
            caption = sample
            if sample[-len(".npy"):] == ".npy":
                data = np.load(f)
            elif sample[-len(".pkl"):] == ".pkl":
                pickle_data = pickle.load(f)
                print(pickle_data.keys())
                data, cap = parse_pickle(pickle_data.copy(), args.pickle_keys)
                caption += cap
                if args.from_humanml:
                    data = torch.from_numpy(data).unsqueeze(0)
                    n_joints = 22 if data.shape[2] == 263 else 21
                    print(f"data.shape={data.shape}")
                    norm_data = data.permute(0, 3, 1, 2) * dataset_torch_std + dataset_torch_mean
                    print(f"norm_data.shape={norm_data.shape}")
                    xyz_data = recover_from_ric(norm_data, n_joints)
                    xyz_data = xyz_data.view(-1, *xyz_data.shape[2:])
                    #zeros = torch.zeros(1, 196, 1, 3)  # Column of zeros
                    #xyz_data_padded = torch.cat([xyz_data, zeros], dim=2)
                    print(f"xyz_data.shape={xyz_data.shape}")
                    data = xyz_data.squeeze(dim=0).numpy() # <<<ls
                print(f"data.shape={data.shape}")
                if args.other_pickle_keys:
                    other_data, cap = parse_pickle(pickle_data.copy(), args.other_pickle_keys)
                    caption += " other:" + cap
            else:
                raise RuntimeError(f"Got non .npy and non .pkl file {sample}")
        
    elif re.findall("\d+", sample) and re.findall("\d+", sample)[0] == sample:
        with open(f"/home/amir.mann/MDM/dataset/HumanML3D/new_joints/{int(sample):06}.npy", "rb") as f:
            data = np.load(f)
        with open(f"/home/amir.mann/MDM/dataset/HumanML3D/texts/{int(sample):06}.txt", "r") as f:
            caption = f.read()
    else:
        print(f"Invalid sample {sample}, which is nither a path to a file nor an integer.")
        exit()
        data = np.array([[[[]]]])
        caption = f"No data found for sample {sample}"
    print(data.shape, other_data.shape if other_data is not None else "No other data")
    return data, caption, other_data


def sample_to_2d(args, motion, rep_i):
    nframes, njoints, _ = motion.shape # [nframes, njoints, 3]
    motion3d_model_shape = torch.from_numpy(motion).permute(1, 2, 0).unsqueeze(0)
    
    #motion_3d (torch.Tensor): Input 3D motion tensor of shape [batch_size, njoints, 3, nframes].
    batch_size = motion3d_model_shape.shape[0]
    if args.random_angles:
        pi = np.pi
        PI = np.pi
        l = eval(args.ver_angle_l)
        u = eval(args.ver_angle_u)
        cam_hor_angles = torch.rand(batch_size) * 2 * np.pi
        cam_ver_angles = (torch.rand(batch_size)) * (u - l) + l
        print(f"randomed hor {cam_hor_angles} ver {cam_ver_angles}")
    else:
        cam_hor_angles = torch.rand(batch_size) * 0 + 2 * np.pi * rep_i / args.num_repetitions
        cam_ver_angles = torch.rand(batch_size) * 0
    cam_distance = torch.ones(batch_size) * args.distance
    cam_shift = compute_into_camera_shift_from_angle(motion3d_model_shape, cam_hor_angles)
    print(compute_into_camera_shift_from_angle(motion3d_model_shape - cam_shift.unsqueeze(1).unsqueeze(3), cam_hor_angles).norm())
    
    motion2d_model_shape, distances = perspective_projection_batch(motion3d_model_shape, cam_hor_angles, cam_ver_angles, cam_distance, cam_shift)
    print("Min distances:", torch.min(distances))
    
    #ret: Tensor of shape [batch_size, njoints, 2, nframes]
    motion2d_plot_shape = motion2d_model_shape.squeeze().permute(2, 0, 1)
    z_dim = torch.zeros(nframes, njoints, 1, device=motion2d_plot_shape.device)  # Shape: [batch_size, njoints, 1, nframes]
    # Concatenate along the third dimension
    motion3d_plot_shape = torch.cat((motion2d_plot_shape, z_dim), dim=2)  # Shape: [batch_size, njoints, 3, nframes]

    return motion3d_plot_shape.numpy()


def save_a_skeleton(args, sample):
    skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained, args.sample_2d, args.date_save)
    
    rep_files = []
    for rep_i in range(args.num_repetitions):
        motion, caption, other_motion = get_data(args, sample)
        if args.sample_2d:
            motion = sample_to_2d(args, motion, rep_i)
            if other_motion is not None:
                other_motion = sample_to_2d(args, other_motion, rep_i)
        try:
            sample_i = int(sample)
        except ValueError:
            if args.pickle_keys:
                sample_i = sum(map(ord, caption))
        save_file = sample_file_template.format(sample_i, rep_i)
        
        animation_save_path = os.path.join(args.output_dir, save_file)
        if rep_i == 0:
            print(sample_print_template.format(caption, sample_i, rep_i, animation_save_path))
        else:
            if not args.sample_2d:
                break
            print(sample_print_template.format("see above", sample_i, rep_i, animation_save_path))
                
        if not args.dont_plot:
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption[:caption.find("#")], fps=20, other_joints=other_motion)
        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
        rep_files.append(animation_save_path)

        #outside_loop
    if not args.dont_plot:
        sample_files = save_multiple_samples(args, args.output_dir,
                                        row_print_template, all_print_template, row_file_template, all_file_template,
                                        caption, num_samples_in_out_file, rep_files, sample_files, sample_i)


def get_argparse_arguments():
    group = argparse.ArgumentParser(prog="Visulize Some Motions")

    group.add_argument("-s", "--samples_to_visualize", required=True,
                       help="Samples To visualize, can be either an integer to use humanML motions range of integers(3-10) or path to pickle / npy file")
    group.add_argument("--output_dir", default='/home/amir.mann/temp', type=str,
                       help="Path to results dir (auto created by the script).")
    group.add_argument("--dont_plot", action='store_true', help="Skips ploting, for debugging.")
    group.add_argument("--date_save", action='store_true', help="Save the animation in a newfile with a different name.")

    group.add_argument("--pickle_keys", nargs="*", default=[],
                       help="Keys to find out of a pickled file, for example --pickle_keys model_output_xyz eval:[7] would get data['model_output_xyz'][[7]].")
    group.add_argument("--other_pickle_keys", nargs="*", default=[],
                       help="Keys to find out of a pickled file as other motion, for example --other_pickle_keys model_output_xyz eval:[7] would get data['model_output_xyz'][[7]].")
    
    group.add_argument("--sample_2d", action="store_true",
                       help="Take the 3d data and sample it into 2d.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action), used when projecting 3d representations into 2d.")
    group.add_argument("--random_angles", action="store_true",
                       help="Pick a random angle for each repitition, instead of using a deterministic sampling from num_repetitions angles.")
    group.add_argument("--ver_angle_u", default="pi/12",
                       help="The vertical angle upper bound to sample from.")
    group.add_argument("--ver_angle_l", default="-pi/12",
                       help="The vertical angle lower bound to sample from.")
    group.add_argument("--distance", default=2.0, type=float,
                       help="Distance between camera and the closest point in the motion on the xz plain")
    group.add_argument("--from_humanml", action='store_true', help="From humanML data representation.")

    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--unconstrained", action='store_true', help="Legacy.")
    group.add_argument("--num_samples", default=10, type=int,help="Legacy.")
    group.add_argument("--guidance_param", default=2.5, type=float, help="Legacy.")
    
    return group.parse_args()


def main():
    
    args = get_argparse_arguments()
    if "-" in args.samples_to_visualize:
        for sample in range(int(args.samples_to_visualize.split("-")[0]), int(args.samples_to_visualize.split("-")[1])):
            print(sample, end=" ")
            save_a_skeleton(args, sample)
    else:
        for sample in args.samples_to_visualize.split("&&"):
            save_a_skeleton(args, sample)

if __name__ == "__main__":
    main()
