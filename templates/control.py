# Author:  Meryll Dindin
# Date:    05 April 2020
# Project: Challenger

import os, json, sys, subprocess

rmf = ['bin', 'etc', 'include', 'lib', 'lib64', 'pyvenv.cfg', 'share']
msk = ['challenger', 'package', 'elementtree', 'ffprobe']

def packages_from_project(path):

    try:
        cmd = 'pipreqs --force --no-pin --print --savepath /dev/null'
        pck = subprocess.check_output(cmd.split(' ') + [path])
        return pck.decode('utf-8')[:-1].split('\n')
    except:
        return []

def update_requirements(path, mask):

    try:
        cmd = 'pipreqs --force --print --savepath /dev/null'
        pck = subprocess.check_output(cmd.split(' ') + [path])
        pck = pck.decode('utf-8')[:-1].split('\n')
        pck = [e for e in pck if not e.split('==')[0] in mask and len(e) > 0]
        if len(pck) != 0: 
            with open('{}/requirements.txt'.format(path), 'w') as f: 
                f.write('\n'.join(pck)+'\n')
    except:
        pass

def compile_list_packages(packages):
    
    lst = []
    for itm in packages: 
        for p in itm:
            if p not in lst:
                lst.append(p)

    return lst

def c_server(file='config-instance.json'):

    cfg = {'instance_type': 't3.micro'}
    if os.path.exists(file): cfg.update(json.load(open(file)))

    return cfg

def get_tags(file='.elasticbeanstalk/config.yml'):

    import yaml

    if os.path.exists(file):
        cfg = yaml.safe_load(open(file))
        app = cfg.get('global').get('application_name').lower()
        try: nme = cfg.get('branch-defaults').get('master').get('environment').lower()
        except: nme = cfg.get('tbranch-defaults').get('default').get('environment').lower()
        cfg = dict(zip(['application', 'service'], [app, nme]))
        return ','.join(['{}={}'.format(k,v) for k,v in cfg.items()])
    else:
        return ''

def env_vars(file='config-environment.json'):

    import datetime

    cfg = dict()
    if os.path.exists(file): cfg.update(json.load(open(file)))
    cfg.update({'BIRTH_DATE': str(datetime.datetime.now().date())})

    return cfg

def get_conf(root='.', file='config-environment.json'):

    cfg = dict()

    for path in os.listdir(root):
        if os.path.isdir('/'.join([root, path])):
            fle = '/'.join([root, path, file])
            if os.path.exists(fle): cfg.update(json.load(open(fle)))

    return cfg

if __name__ == '__main__':

    if sys.argv[1] == 'config-project':

        os.system('rm -rf {}'.format(' '.join(rmf)))
        os.system('python3 -m venv .')

    if sys.argv[1] == 'create-project':

        os.system('pip install setuptools wheel pip --upgrade')
        os.system('pip install pipreqs --upgrade')
        frc = ['numpy', 'cmake', 'jupyter', 'notebook', 'ipython', 'ipykernel']
        with open('requirements.txt', 'w') as f: f.write('\n'.join(frc)+'\n')
        os.system('pip install -r requirements.txt')
        os.system('pip install jupyter_contrib_nbextensions')
        os.system('jupyter contrib nbextension install --user')
        os.system('jupyter nbextension enable codefolding/main')
        src = os.getcwd().split('/')[-1]
        os.system('python -m ipykernel install --user --name={}'.format(src.lower()))
        lst = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith('.') and d not in rmf]
        lst = [packages_from_project(d) for d in lst]
        lst = compile_list_packages(lst)
        lst = [p for p in lst if p not in msk + frc]
        with open('requirements.txt', 'w') as f: f.write('\n'.join(lst)+'\n')
        os.system('pip install -r requirements.txt')
        os.remove('requirements.txt')

    if sys.argv[1] == 'update-project':

        lst = [d for d in os.listdir() if os.path.isdir(d) and not d.startswith('.') and d not in rmf]
        for drc in lst: update_requirements(drc, msk)

    if sys.argv[1] == 'config-python':

        vars_env = env_vars()
        cfg_size = len(vars_env.keys())

        if not os.path.exists('bin/activate-origin'):
            os.system('cp bin/activate bin/activate-origin')
        os.system('cp bin/activate-origin bin/activate')

        env_vars = get_conf()
        add_vars = ['export {}={}'.format(key, env_vars.get(key)) for key in sorted(env_vars.keys())]
        add_vars = '\n' + '\n'.join(add_vars)
        del_vars = ['unset {}'.format(key) for key in sorted(env_vars.keys())]
        del_vars = '\n    ' + '\n    '.join(del_vars)

        old_file = open('bin/activate').readlines()
        new_file = ''.join(old_file[:37]) + del_vars + ''.join(old_file[37:]) + add_vars
        open('bin/activate', 'w').write(new_file)

    if sys.argv[1] == 'create-service':

        src_tags = get_tags()
        vars_env = env_vars()
        cfg_size = len(vars_env.keys())
        c_server = c_server()

        template = 'eb create {} {} --envvars {} --tags {}'
        instance = ' '.join(["--{} '{}'".format(k, str(v)) for k,v in c_server.items()])
        env_vars = ','.join(["{}='{}'".format(k, str(v)) for k,v in vars_env.items()])

        print('\n# Launch {} Creation'.format(sys.argv[2]))
        print('# On {} with {} Associated Variables\n'.format(c_server.get('instance_type'), cfg_size))
        os.system(template.format(sys.argv[2], instance, env_vars, src_tags))

    if sys.argv[1] == 'config-service':

        vars_env = env_vars()
        cfg_size = len(vars_env.keys())

        template = 'eb setenv {}'
        env_vars = ','.join(['='.join([k, str(v)]) for k,v in vars_env.items()])

        print('\n# Update Environment Variables')
        print('# {} Associated Variables\n'.format(cfg_size))
        os.system(template.format(env_vars))

    if sys.argv[1] == 'config-docker':

        vars_env = env_vars()
        cfg_size = len(vars_env.keys())

        env_vars = '\n'.join(['='.join([k, str(v)]) for k,v in vars_env.items()])
        env_file = open('config-docker.env', 'w')
        env_file.write(env_vars + '\n')

    if sys.argv[1] == 'config-lambda':

        sip = sys.argv[2].replace('.', '-')
        try: avz = sys.argv[3]
        except: avz = 'us-east-2'
        try: key = sys.argv[4]
        except: key = '../aws.pem'

        os.system("scp -i {} packages.sh requirements.txt ec2-user@ec2-{}.{}.compute.amazonaws.com:~".format(key, sip, avz))
        os.system("ssh -i {} ec2-user@ec2-{}.{}.compute.amazonaws.com 'bash -s' < packages.sh".format(key, sip, avz))
        os.system("scp -i {} ec2-user@ec2-{}.{}.compute.amazonaws.com:~/app/packages.zip .".format(key, sip, avz))
        os.system("unzip packages.zip -d packages")
        os.remove("packages.zip")

    if sys.argv[1] == 'create-lambda':

        os.system("mkdir tmp; cp *.py *.json tmp; cp -r packages/* tmp")
        os.system("cd tmp; zip -r ../function.zip *; cd ..; rm -rf tmp")
