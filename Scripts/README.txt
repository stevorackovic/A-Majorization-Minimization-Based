Scripts used to run our experiments with MetaHumans:
    extract_from_maya.py - it is executed in Autodesk Maya and extracts the blendshapes and corrective blendshapes;
        preprocessing.py - subsamples the vertices (reduces the original number from more than 20k to 4k) excluding those in the neck and shoulders region as well as the least active ones;
              MM_back.py - contains the functions applied in our algorithm that are then called in the script for execution;
  CubicEquationSolver.py - contains a function for fast computation of roots of a cubic eq
              execute.py - the scrip for execution takes pseudoinverse initialization and runs our algorithm for different values of the regularization parameter.

