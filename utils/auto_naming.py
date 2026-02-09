import os


def get_exp_name(args):
    args.save_dir = ""
    grd = ""
    grd += args.selection_method if args.selection_method != "none" else ""
    folder = f"/{args.dataset}"
    args.save_dir += f"{folder}_{args.arch}_lr{args.lr}"
    args.save_dir += f"_corrupt{args.corrupt_ratio}" if args.corrupt_ratio > 0 else ""
    subset_size = args.train_frac
    args.save_dir += f"_train{subset_size:.2f}"
    if args.random_subset_size < 1.0:
        args.save_dir += (
            f"_random{args.random_subset_size:.2f}-start{args.partition_start}"
        )
    grd += (
        f"_dropevery{args.drop_interval}-loss{args.drop_thresh}-watch{args.watch_interval}"
        if args.drop_learned
        else ""
    )
    # args.save_dir += f"_batchsize{args.batch_size}_{grd}"

    if args.selection_method == "crest":
        args.save_dir += "_coreset" if args.approx_with_coreset else "_subset"
        args.save_dir += "_momentum" if args.approx_moment else ""
        grd += f"-batchnummul{args.batch_num_mul}-interalmul{args.interval_mul}"
        grd += f"_thresh-factor{args.check_thresh_factor}"
    if args.selection_method == "ensemble":
        args.save_dir += f"_ensemble_selection_method-{args.selection_method_ensemble}"
        args.save_dir += f"_ensemble_size-{args.ensemble_num}"
        args.save_dir += (
            f"_update_separate-{args.update_separate}" if args.update_separate else ""
        )
        args.save_dir += (
            f"_use_same_initialization-{args.use_same_initialization}"
            if args.use_same_initialization
            else ""
        )
    if args.selection_method == "single_spread":
        args.save_dir += f"noise_std-{args.noise_std}"
    if args.selection_method == "single_spread_bn":
        args.save_dir += f"noise_std_bn-{args.noise_std}"

    args.save_dir += f"_seed_{args.seed}"
    args.save_dir = os.getcwd() + args.save_dir
    return args.save_dir
