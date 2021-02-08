### Step 1: Check if Java is already installed

```shell script
java -version
```

If Java is not installed, you will see something like:

```shell script
Output
Command 'java' not found, but can be installed with:

apt install default-jre
apt install openjdk-11-jre-headless
apt install openjdk-8-jre-headless
```

You may simply install the default from OpenJDK:

```shell script
sudo apt update
sudo apt install default-jre
```

or 

```shell script
sudo apt install default-jdk
```

Afterwards, it is good to specify `JAVA_HOME`:

```shell script
sudo nano /etc/environment
```

Then add it in the next line:
```shell script
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"
JAVA_HOME=/usr/lib/jvm/default-java
```

Verify it:
```shell script
source /etc/environment
echo $JAVA_HOME
```
