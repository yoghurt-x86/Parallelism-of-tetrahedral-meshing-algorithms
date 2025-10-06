{
  description = "Combined libigl + HIP development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in
      {
        devShells = {
          # Default shell with libigl dependencies only
          default = pkgs.mkShell {
            packages = with pkgs; [
              # Development tools
              clang-tools
              cmake
              codespell
              conan
              cppcheck
              doxygen
              gtest
              lcov
              vcpkg
              vcpkg-tool

              # libigl dependencies
              glfw
              xorg.libX11
              xorg.libXrandr
              xorg.libXinerama
              xorg.libXcursor
              xorg.libXi

              boost
              gmp
              mpfr

            ] ++ (if system == "aarch64-darwin" then [ ] else [ gdb ]);

            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH"
              export CMAKE_PREFIX_PATH="${pkgs.boost.dev}:${pkgs.boost}:$CMAKE_PREFIX_PATH"
              echo "X11 libraries configured. If running on Wayland, start Xwayland with 'Xwayland :10' and set 'export DISPLAY=:10'"
              echo "Standard libigl environment loaded"
              echo "Use 'nix develop .#hip' for GPU acceleration support"
            '';
          };

          # HIP-enabled shell with GPU acceleration
          hip = pkgs.mkShell {
            packages = with pkgs; [
              # Development tools (same as default)
              clang-tools
              cmake
              codespell
              conan
              cppcheck
              doxygen
              gtest
              lcov
              vcpkg
              vcpkg-tool

              # libigl dependencies
              glfw
              xorg.libX11
              xorg.libXrandr
              xorg.libXinerama
              xorg.libXcursor
              xorg.libXi

              boost
              gmp
              mpfr


              # ROCm packages for HIP/GPU support
              rocmPackages.hipcc
              rocmPackages.clr
              rocmPackages.clr.icd
              rocmPackages.rocm-runtime
              rocmPackages.rocminfo
              rocmPackages.rocm-device-libs

              # Build tools
              gnumake
              gcc
              pkg-config
            ] ++ (if system == "aarch64-darwin" then [ ] else [ gdb ]);

            shellHook = ''
              # Standard libigl environment
              export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH"
              export CMAKE_PREFIX_PATH="${pkgs.boost.dev}:${pkgs.boost}:$CMAKE_PREFIX_PATH"
              
              # HIP/ROCm environment
              export HIP_PATH=${pkgs.rocmPackages.clr}
              export HIP_PLATFORM=amd
              export ROC_ENABLE_PRE_VEGA=1
              export PATH=${pkgs.rocmPackages.hipcc}/bin:$PATH
              
              # Create temporary ROCm structure that hipcc expects
              export TMPDIR_ROCM=$(mktemp -d)
              mkdir -p $TMPDIR_ROCM/opt/rocm/{bin,lib/llvm/bin}
              ln -sf ${pkgs.rocmPackages.clr}/bin/rocm_agent_enumerator $TMPDIR_ROCM/opt/rocm/bin/
              ln -sf ${pkgs.rocmPackages.hipcc}/bin/clang++ $TMPDIR_ROCM/opt/rocm/lib/llvm/bin/
              
              # Add the temp ROCm to PATH so hipcc can find the tools
              export PATH=$TMPDIR_ROCM/opt/rocm/bin:$TMPDIR_ROCM/opt/rocm/lib/llvm/bin:$PATH
              
              echo "X11 libraries configured. If running on Wayland, start Xwayland with 'Xwayland :10' and set 'export DISPLAY=:10'"
              echo "Combined libigl + HIP development environment loaded"
              echo "ROCm temporary directory: $TMPDIR_ROCM"
              echo "GPU acceleration enabled!"
              
              # Cleanup function
              cleanup() {
                rm -rf $TMPDIR_ROCM
              }
              trap cleanup EXIT
            '';

            # Additional environment variables for HIP
            HIP_VISIBLE_DEVICES = "0";
            AMD_LOG_LEVEL = "3";
          };
        };
      });
}
