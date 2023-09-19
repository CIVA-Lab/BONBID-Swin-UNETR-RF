# Deep and Local Features Segmentation Algorithm Docker
Prepared by Imad Toubal, 9/20/2023

## Before cloning
This project uses Git Large File Storage (LFS) to handle large files. To work with this repository properly, you'll need to have Git LFS installed and set up. Here's how:

### 1. Install Git LFS

First, ensure you have Git LFS installed. If not, download and install it from [https://git-lfs.github.com/](https://git-lfs.github.com/).

### 2. Clone the Repository

If you haven't cloned the repository yet, you can do so with:

```bash
git clone <repository-url>
```

If you've already cloned the repository before installing Git LFS, navigate to your repository and pull the LFS files:

```bash
git lfs install
git lfs pull
```

## Run with Docker

Before running, you will need a local docker installation.
For more details, please read grand-challenge documents [https://grand-challenge.org/documentation/automated-evaluation/](https://grand-challenge.org/documentation/creating-an-algorithm-container/) and [https://comic.github.io/evalutils/usage.html](https://comic.github.io/evalutils/usage.html#algorithm-container) 

predict.py is the main function for generating prediction results.

Noted that when your algorithm is run on grand challenge `<local_path>/case1`
will be mapped to `/input`. Then a separate run will be made with
`<local_path>/case2` mapped to `/input`. This allows grand challenge to execute
all the jobs in parallel on their cloud infrastructure. For simplicity, you can
include one case in your test data when you test locally. The platform will
handle multiple cases. Predict should only handle one case.


Please follow these steps to run it on the local machine.


1. Build the docker
  ```console
   ./build.sh
  ```
1. Test the docker
  ```console
  ./test.sh
  ```

In test.sh, use the following command in order to generate the results locally
and test your codes:

```console
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v $SCRIPTPATH/output/:/output/ \
        bondbidhie2023_algorithm
```
But for uploading algorithm docker to the grand challenge server, please use the codes that I provided in test.sh.

```console
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v bondbidhie2023_algorithm-output-$VOLUME_SUFFIX:/output/ \
        bondbidhie2023_algorithm
```
3. Exporting docker ./export.sh 
Running ./export.sh, and submitting the generated zip file of the algorithm docker.

## References
1. Original repository by [Rina Bao](https://github.com/baorina): https://github.com/baorina/BONBID-HIE-MICCAI2023
2. [Challanege website](https://bonbid-hie2023.grand-challenge.org/bonbid-hie2023/)
3. [Dataset download](https://zenodo.org/record/8104103)
4. [Preprint challenge paper](https://www.biorxiv.org/content/10.1101/2023.06.30.546841v1.abstract)
5. [MONAI](https://github.com/Project-MONAI/MONAI)
6. [Hausdorff Loss](https://arxiv.org/abs/1904.10030)
