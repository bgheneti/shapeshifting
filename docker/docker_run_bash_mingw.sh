#! /bin/bash
echo "Drake Docker container for WINDOWS (via MinGW / Docker Toolbox)"
if [ "$#" != "3" ]; then 
	echo "Please supply three arguments: a Drake release (drake-YYYYMMDD), a relative path to a directory to mount as /notebooks, and your computer's IP address (from ipconfig)."
	exit 1
else
	docker pull bgheneti/shapeshift:$1
	docker run -it  -e DISPLAY=$3:0 --rm \
			        -v "$(pwd)/$2":/notebooks -p 7000-7010:7000-7010 bgheneti/shapeshift:$1 \
			        /bin/bash -c "cd /notebooks && /bin/bash"
fi
