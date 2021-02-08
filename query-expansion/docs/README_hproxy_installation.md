## Install the HTTP proxy - HProxy 
* Download the HProxy code to a directory by 
`wget https://rnd-gitlab-eu.huawei.com/RndTool/Hproxy/-/archive/master/Hproxy-master.zip --no-check-certificate --no-proxy` 
* Unzip the downloaded file `unzip Hproxy-master.zip && mv Hproxy-master hproxy`
* 
```
cd linux
sudo ./hproxy_setup.sh
```
You can also check the original refernce [here](https://rnd-gitlab-eu.huawei.com/IT-Navigator/Main/-/wikis/HProxy).

* When choosing proxy servers, type `i` to input your own proxy:
    `proxyde.huawei.com`, and use your Huawei domain account (NOT w3 account) to log in.
    
