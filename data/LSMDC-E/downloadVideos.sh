#!/bin/bash

EXPECTED_ARGS=2
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` <username-MPIIMD> <password-MPIIMD> [parallelDownloads=8]"
  exit $E_BADARGS
fi
usernameMD=$1
passwordMD=$2
parallelDownloads=$3

if [ $# -lt 3 ]
then
  parallelDownloads=8
fi

########## download videos
# avi clips from MPII-MD
wget http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/MPIIMD_downloadLinks.txt --user=$usernameMD --password=$passwordMD
filesToDownloadMD="MPIIMD_downloadLinks.txt"
cat $filesToDownloadMD | xargs -n 1 -P $parallelDownloads wget -crnH --cut-dirs=3 -q --user=$usernameMD --password=$passwordMD

# avi clips from M-VAD-test-aligned
wget http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/MVADaligned_downloadLinks.txt --user=$usernameMD --password=$passwordMD
filesToDownloadMVADTEST="MVADaligned_downloadLinks.txt"
cat $filesToDownloadMVADTEST | xargs -n 1 -P $parallelDownloads wget -crnH --cut-dirs=3 -q --user=$usernameMD --password=$passwordMD
