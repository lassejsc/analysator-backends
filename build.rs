use std::path::Path;
use std::process::Command;
use std::{env, path::PathBuf};

fn main() {
    let zfp_dst = cmake::Config::new("external/zfp").build();
    let eigen_dst = cmake::Config::new("external/eigen")
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build();

    let octree_src = Path::new("external/tucker-octree");
    let cpp_file = octree_src.join("toctree.cpp");
    Command::new("sed")
        .args(&["-i", "s/ColMajor/RowMajor/g", cpp_file.to_str().unwrap()])
        .status()
        .expect("Failed to modify toctree.cpp (RowMajor)");

    Command::new("sed")
        .args(&[
            "-i",
            "s/Eigen::Vector<T, Eigen::Dynamic>/Eigen::Matrix<T, Eigen::Dynamic, 1>/g",
            cpp_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to modify toctree.cpp (Eigen Fix)");

    let octree_dst = cmake::Config::new(octree_src)
        .define("TOCTREE_L2ERROR", "true")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("zfp_DIR", format!("{}/lib/cmake/zfp", zfp_dst.display()))
        .define(
            "Eigen3_DIR",
            format!("{}/share/eigen3/cmake", eigen_dst.display()),
        )
        .build();

    for cfg in ["no_nn", "no_octree"] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
    println!("cargo:rerun-if-env-changed=MLP_COMPRESSION_DIR");
    let linux = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");

    link_so(
        "MLP_COMPRESSION_DIR",
        "libvlasiator_vdf_compressor_nn.so",
        "no_nn",
        linux,
        |_dir| println!("cargo:rustc-link-lib=vlasiator_vdf_compressor_nn"),
    );

    let octree_lib_dir = octree_dst.join("lib");
    let zfp_lib_dir = zfp_dst.join("lib");
    println!(
        "cargo:rustc-link-search=native={}",
        octree_lib_dir.display()
    );
    println!("cargo:rustc-link-search=native={}", zfp_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=toctree_compressor");
    println!("cargo:rustc-link-lib=dylib=zfp");
    println!("cargo:rustc-link-lib=stdc++");

    if linux {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            octree_lib_dir.display()
        );
    }
    println!("cargo:rerun-if-changed=external/tucker-octree/toctree.cpp");
}

fn link_so(
    env_key: &str,
    so_name: &str,
    cfg_if_missing: &str,
    linux: bool,
    emit_libs: impl FnOnce(&PathBuf),
) {
    let Some(base) = env::var_os(env_key).map(PathBuf::from) else {
        println!("cargo:rustc-cfg={cfg_if_missing}");
        return;
    };

    let candidates = [
        base.join("lib").join(so_name),
        base.join(so_name),
        base.join("build").join(so_name),
    ];

    let Some(so) = candidates.into_iter().find(|p| p.exists()) else {
        println!("cargo:rustc-cfg={cfg_if_missing}");
        return;
    };

    let dir = so.parent().unwrap().to_path_buf();
    println!("cargo:rustc-link-search=native={}", dir.display());
    emit_libs(&dir);

    if linux {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rerun-if-changed={}", so.display());
}
