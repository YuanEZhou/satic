#!/bin/sh

cp -r $1   $2
cd $2
mv infos_$1-best.pkl infos_$2-best.pkl 
mv infos_$1.pkl infos_$2.pkl 
cd ../