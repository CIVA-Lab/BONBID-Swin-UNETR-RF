# bondbidhie2023_algorithm Algorithm Docker
Prepared by Rina Bao, 7/12/2023

The source code for the algorithm container for
bondbidhie2023_algorithm, generated with
evalutils version 0.4.2
using Python 3.8.


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
2. Test the docker
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
