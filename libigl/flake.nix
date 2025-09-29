{
  description = "A Nix-flake-based C/C++ development environment";

  #inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs =
    inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs { inherit system; config.allowUnfree = true; };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default =
            pkgs.mkShell.override
              {
                # Override stdenv in order to change compiler:
                # stdenv = pkgs.clangStdenv;
              }
              {
                packages =
                  with pkgs;
                  [
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
#                    vscode

		    # libigl dependencies:
		    glfw
		    xorg.libX11
		    xorg.libXrandr
		    xorg.libXinerama
		    xorg.libXcursor
		    xorg.libXi

		    boost
		    gmp
		    mpfr

            jetbrains.clion
		  ]
                  ++ (if system == "aarch64-darwin" then [ ] else [ gdb ]);

                # Set up library paths for X11 libraries
                shellHook = ''
                  export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH"
                  export CMAKE_PREFIX_PATH="${pkgs.boost.dev}:${pkgs.boost}:$CMAKE_PREFIX_PATH"
                  echo "X11 libraries configured. If running on Wayland, start Xwayland with 'Xwayland :10' and set 'export DISPLAY=:10'"
                '';
              };
        }
      );
    };
}
