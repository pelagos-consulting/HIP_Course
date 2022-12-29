#!/bin/bash

fname=users.csv
fname_export=users_export.csv
rm $fname
rm $fname_export

let count=0
for n in $(seq -f "%02g" 1 99)
do
    pass=$(gpw 1)
    usr="lab$n"

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

    # Create the user and add to groups
    groupadd jupyter-$usr --gid $((3000+count))
    adduser jupyter-$usr --disabled-password --gecos "" --uid $((3000+count)) --gid $((3000+count))

    adduser $usr video
    adduser $usr render
    
    adduser jupyter-$usr video
    adduser jupyter-$usr render

    #change password for user
    echo "$usr:$pass" >> $fname
    echo "$usr:$pass" >> $fname_export   
    echo "jupyter-$usr:$pass" >> $fname
    let count++
done

# Batch change user passwords
chpasswd < $fname
