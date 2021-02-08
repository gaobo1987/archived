##  Setting up remote Python interpreters
Bo Gao, July 21, 2020

#### Prerequisites: 
* You have access to a remote Linux machine, for example,
    one of our instances in Frankfurt, 
    which has Ubuntu 18.04 pre-installed. For access to this machine, 
    please ask Robin Cahierre WX920149 or Bo Gao b00563677.
* Basic understanding of Python and Python virtual environment.
* BTW, check your CPU specs by `lscpu`, the RAM specs by `free -h` 
and the GPU status by `nvidia-smi` (if it is installed).

#### Step 1: Set up the host file if you haven't (with root):
```shell script
# add the host to /etc/hosts
hostn=$(cat /etc/hostname)
sudo echo '127.0.0.1    '$hostn >>/etc/hosts
```

#### Step 2: Generate SSH keys if you haven't
`ssh-keygen -t rsa -C "your_email@example.com"`


#### Step 3: Install the HTTP proxy - HProxy 
* Download the HProxy code to a directory by 
`wget https://rnd-gitlab-eu.huawei.com/RndTool/Hproxy/-/archive/master/Hproxy-master.zip --no-check-certificate --no-proxy` 
* Unzip the downloaded file `unzip Hproxy-master.zip && mv Hproxy-master hproxy`
* Follow the instructions [here](https://rnd-gitlab-eu.huawei.com/IT-Navigator/Main/-/wikis/HProxy) 
    to install HProxy
* When choosing proxy servers, type `i` to input your own proxy:
    `proxyde.huawei.com`, and use your Huawei domain account (NOT w3 account) to log in.
    
#### Step 4: Install [pyenv](https://github.com/pyenv/pyenv) for Python version control:
* Always good to update: 
    ```
    sudo apt update
    sudo apt upgrade
    ```
* Install pyenv dependencies: 
    ```shell script
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev \
    wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
    ```
* Clone the pyenv git repo to home directory:
`git clone https://github.com/pyenv/pyenv.git ~/.pyenv`
* Configure the environment variables:
    ```shell script
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
    ```
* Restart bash shell: `exec bash`
* Verify installation: `pyenv versions`
* Install a new python version: `pyenv install 3.7.8`, 
    this could take a while since it builds from source.
* Do `pyenv versions` again, the expected output is:
    ```shell script
    $ pyenv versions
    * system (set by /root/.pyenv/version)
      3.7.8
    ```
* Switch to a different version globally: 
    ```shell script
    $ pyenv global 3.7.8
    $ python --version
    Python 3.7.8
    ```
  
#### Step 5: Create Virtual Environments
* Create an virtual environment and install libraries as you like, e.g.
    ```shell script
    $ python -m venv venv
    $ source venv/bin/activate
    (venv) $ pip install some-package 
    ```

#### Step 6: Configure PyCharm to the remote Interpreter
* If you haven't already, download and install the latest version of 
    PyCharm Professional edition.
* Go to "File > Settings", find the "Python Interpreter" interface
* Click the gear button to "Add" a new interpreter
* In the popup window, choose "SSH Interpreter" on the left panel
* Choose "New server configuration"
* Fill out the server information, which we assume you already have
* Choose the python path from the virtual environment you just created
* Choose the project directory you just created as the sync folder
* Click "Finish"


Voila, now you have set up the remote python interpreter for your project!
You can now freely install python libraries in the remote machine. 
Project code will first be synced to the remote folder, executed there,
and the results are then resturned/synced.
