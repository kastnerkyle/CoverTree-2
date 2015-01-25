with import <nixpkgs> {};

let
  joblib = callPackage ./nix/joblib.nix {};

  pymodules = with python27Packages; [
    python27
    ipython
    numpy
    scipy
    matplotlib
    pandas

    joblib
  ];

  deps = pymodules ++ [
    graphviz
  ];

in

pkgs.myEnvFun {
  name = "covertree";
  buildInputs = deps;
}