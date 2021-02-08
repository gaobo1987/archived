### Download & install docker
[Optional] Uninstall previous versions:
```bash
sudo apt-get remove docker docker-engine docker.io docker-ce
```
Update all packages and install requirements:
```bash
sudo apt update
```
```bash
sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common
```

Add the GPG key for the official Docker repository to your system:
```bash
curl -fsSL http://mirrors.tools.huawei.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
```
Add the docker repo to the APT sources & update package db:
```bash
sudo add-apt-repository "deb [arch=amd64] http://mirrors.tools.huawei.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
```
```bash
sudo apt update
```

Install docker
```bash
sudo apt install docker-ce
```

### Add users to the docker group [optional]
By default you need to run docker as sudo. If you add yourself and other users to the docker group then the sudo command can be dropped.

Add yourself:
```bash
sudo usermod -aG docker ${USER}
```

Add other users:
```bash
sudo usermod -aG docker <username>
```
**Note:** you need to re-login to enable your new privileges. Either just restart the terminal or run: `su - ${USER}`

### Download & install docker-compose

If there is a more recent version of docker-compose then replace the version below. You can check it [here](https://github.com/docker/compose/releases/).

```bash
sudo wget https://github.com/docker/compose/releases/download/1.26.2/docker-compose-Linux-x86_64 -O /usr/local/bin/docker-compose
```

Set the permissions:

```bash
sudo chmod +x /usr/local/bin/docker-compose
```

Verify the installation:

```bash
docker-compose --version
```


### Re-run the HProxy setup script
If you already installed and run the HProxy script, then you need to rerun it for the proxy to be configured correctly.
* `cd hproxy/linux`
* `sudo ./hproxy_setup.sh`
* When choosing proxy servers, type `i` to input your own proxy: `proxyde.huawei.com`, and use your Huawei domain account (NOT w3 account) to log in.


### Remarks:
* We ran into an issue with the proxy after running a docker-compose command. This is because by default the created subnet overlaps with the internal subnet of the huawei cloud service. To solve this issue you need to specify the network and subnet. here is an example that worked for us:
```yaml
networks:
  qa_network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 181.42.0.1/16
```
