#! /bin/bash
ip=8080
if [ "$#" != "2" ]; then 
	echo "Please supply two arguments: a Drake release (drake-YYYYMMDD), and a relative path to your preferred notebook directory"
	exit 1
else
	echo -e "\033[32m" Using $(pwd)$2 as your notebook root directory. "\033[0m"	
	#docker pull shapeshift:$1
  docker run -it -e GRB_LICENSE_FILE='/notebooks/gurobi.lic' -p $ip:8080 -p 7000-7010:7000-7010 --rm -v "$(pwd)/$2"":"/notebooks \
		bgheneti/shapeshift:$1 /bin/bash -c "cd /notebooks && jupyter notebook --ip 0.0.0.0 --port 8080 --allow-root --no-browser"
fi
