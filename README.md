## A 418 Final Project by Junye(Hans) Chen and Zhan(Patrick) Dong

[Proposal](https://github.com/patrickzhandong/ParallelPlateRecoginition/blob/master/proposal.pdf)
[Checkpoint](https://github.com/patrickzhandong/ParallelPlateRecoginition/blob/master/CheckpointReport.pdf)

# Summary
We want to write a parallel program for license plate number recognition,
a very interesting topic in the eld of computer vision. The process of plate number recognition
involves various computer vision algorithms such as image segmentation, edge detection, etc. We
will use CUDA (and maybe OpenMP), and try to have a good speedup against the benchmarks.

# Schedule
Nov.20 - Nov. 23: Continue working on the implementation for character segmentation, improve its accuracy. Working on the minor steps after edge detection to find the plate region. 

Nov.24 - Nov. 26: Working on character recognition with existing library, start parallelizing edge detection.Continue working on the minor steps. Start parallelizing the preprocessing step.

Nov.27 - Nov. 30: Continue working on parallelizing edge detection.Continue parallelizing the preprocessing step.

Dec.1 - Dec. 3: Start parallelizing the segmentation step.Continue parallelizing the segmentation step.Trying different preprocessing techniques to improve the accuracy. Start working on character recognition (If have time).

Dec.4 - Dec.7:Continue working on the work undone.Working  on character recognition(If have time).

Dec.8 - Dec.12: Wrap up the algorithm.Do the speed tests.Write the Report.


