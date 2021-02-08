##  Run a Jupyter notebook remotely and access it locally
Bo Gao, July 28, 2020

#### Prerequisites: 
* You have access to a remote Linux machine, for example,
    one of our instances in Frankfurt, 
    which has Ubuntu 18.04 pre-installed. For access to this machine, 
    please ask Robin Cahierre WX920149 or Bo Gao b00563677.
* Basic understanding of Python, Python virtual environment
    and Jupyter notebooks.

#### Step 1: Run Jupyter notebook from remote machine:
It is recommended to run the notebook 
within a virtual environment on you remote machine. 
If you haven't installed notebook, install it:
```shell script
(venv) $ pip install notebook
```
Give your remote notebook a port number of your choice.
```shell script
(venv) $ jupyter notebook --no-browser --port=XXXX
```
e.g. 
```shell script
(venv) $ jupyter notebook --no-browser --port=1234
```

Alternative: directly specify the remote IP
```shell script
(venv) $ jupyter notebook --no-browser --ip=xx.xx.xx.xx --port=XXXX
```
e.g. 
```shell script
(venv) $ jupyter notebook --no-browser --ip=10.206.41.17 --port=1234
```

#### Step 2: Forward port XXXX to YYYY and listen to it
On you local terminal, do
```shell script
ssh -N -f -L localhost:YYYY:localhost:XXXX remoteuser@remotehost
```
e.g.
```shell script
ssh -N -f -L localhost:1234:localhost:1234 b00563677@10.206.41.17
```

* ssh: your handy ssh command. See man page for more info
* -N: suppresses the execution of a remote command. 
Pretty much used in port forwarding.
* -f: this requests the ssh command to go to background before execution.
* -L: this argument requires an input in the form of 
local_socket:remote_socket. 
Here, we’re specifying our port as YYYY 
which will be binded to the port XXXX 
from your remote connection.

Remark: when the notebook is run with explicitly specified IP, the localhost to localhost forward wouldn't work

#### Step 3: Fire-up Jupyter Notebook
Copy and paste the notebook URL printed 
in the remote machine terminal, with its token,
into your local browser
`http://localhost:YYYY/?token=...`

Now you should have the notebook running locally, of which
the interpreter is sitting in the remote.
