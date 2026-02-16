# HyperSpectral
## Description

Application for viewing and processing hyperspectral images saved in ENVI format.

The project was originally created for my engineering project, but I have continued working on it ever since.

Originally, I used Dear ImGui as the GUI framework, but I decided to migrate to Qt (currently on the *refactor_image* branch).

I have also decided to rewrite the GPU processing pipeline in the form of a task graph, but this is still a work in progress on the *refactor_image* branch.

## Installation

```
mkdir build
cd build
cmake ..
cmake --build .
```


External libraries are managed through CMake FetchContent. The only exception is the Qt6 library, which must be installed separately, required only for *refactor_image* branch.
