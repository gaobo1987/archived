## Check basic stats:

#### check memory:
`free -m`

`vmstat -s`

#### check disks:
`lsblk` and `df`


#### check gpu:
* `nvidia-smi`
* `sudo lshw -C display`


#### check cpu:
`lscpu`


#### check ubuntu version:
`lsb_release -a`


## Use screen
* activate an unnamed screen session: `screen` + SPACE
* activate a named screen session: `screen -S SESSION_NAME`
* detach from current screen: CTL+A, then press D
* show list of screens: `screen -ls`
* restore a screen: `screen -r SESSION_ID`


## Extend LVMs on Linux
[useful reference](https://www.howtogeek.com/howto/40702/how-to-manage-and-use-lvm-logical-volume-management-in-ubuntu/)

1. `sudo lvextend -L+3G /PATH/TO/LVM`
2. `sudo resize2fs /PATH/TO/LVM`


## Add or delete user
see [here](https://www.digitalocean.com/community/tutorials/how-to-add-and-delete-users-on-ubuntu-16-04)

## Intall Nvidia driver and cuda
see [here](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07)


