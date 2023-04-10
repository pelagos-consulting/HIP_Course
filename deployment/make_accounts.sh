#!/bin/bash

# Input files
fname=$(mktemp)
fname_export=users.csv

let count=0

cat ${fname_export} | while read line
do
    usr=$(echo $line | cut -d ':' -f 1)
    pass=$(echo $line | cut -d ':' -f 2)

    # Check if user exists
    if id -u "$usr" > /dev/null 1>&1
    then
        # User exists
        userdel -rf $usr
	groupdel $usr
    fi 
    
    # Add the user back
    groupadd $usr --gid $((2000+count))
    adduser $usr --disabled-password --gecos "" --uid $((2000+count)) --gid $((2000+count))

    # Check if jupyter-user exists
    if id -u "jupyter-$usr" > /dev/null 1>&1
    then
        userdel -rf jupyter-$usr
	groupdel jupyter-$usr
    fi

    # Create the jupyter user and add to groups
    groupadd jupyter-$usr --gid $((3000+count))
    adduser jupyter-$usr --disabled-password --gecos "" --uid $((3000+count)) --gid $((3000+count))

    adduser $usr video
    adduser $usr render
    
    adduser jupyter-$usr video
    adduser jupyter-$usr render

    echo "$usr:$pass" >> $fname
    echo "jupyter-$usr:$pass" >> $fname

    # Increment counter
    let count++
done

# Batch change user passwords
chpasswd < $fname
rm -rf $fname
