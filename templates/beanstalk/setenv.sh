#!/bin/bash

# Author:  DINDIN Meryll
# Date:    23 Octobre 2019
# Project: beanstalk

# Publish environment variable

if [ "$1" == "dev" ]; then

    python -c "import os, json;\
    cfg=json.load(open('environment.json'));\
    cmd='\n'.join(['='.join(['export {}'.format(k),str(v)]) for k,v in cfg.items()]);\
    print('# Write chunk at the end of bin/activate');\
    print(cmd + '\n');\
    cmd='\n'.join(['unset {}'.format(k) for k in cfg.keys()]);\
    print('# Insert chunk inside deactivate function');\
    print(cmd);"

elif [ "$1" == "prod" ]; then 

    python -c "import os, json;\
    cfg=json.load(open('environment.json'));\
    cmd=' '.join(['='.join([k,str(v)]) for k,v in cfg.items()]);\
    os.system('eb setenv {}'.format(cmd))"

fi