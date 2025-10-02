{
  description = "HIP development environment for task3";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # ROCm packages - minimal set
            rocmPackages.hipcc
            rocmPackages.clr
            rocmPackages.clr.icd
            rocmPackages.rocm-runtime
            rocmPackages.rocminfo
            rocmPackages.rocm-device-libs

            # Build tools
            gnumake
            gcc

            # Utilities
            pkg-config
          ];
          
          # Environment variables for ROCm
          shellHook = ''
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
            
            echo "HIP development environment loaded"
            echo "ROCm temporary directory: $TMPDIR_ROCM"
            echo "Run 'make' to build the project"
            
            # Cleanup function
            cleanup() {
              rm -rf $TMPDIR_ROCM
            }
            trap cleanup EXIT
          '';
          
          # Additional environment variables
          HIP_VISIBLE_DEVICES = "0";
          AMD_LOG_LEVEL = "3";
        };
      });
}
