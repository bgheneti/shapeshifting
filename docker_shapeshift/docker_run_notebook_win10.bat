@ECHO "Drake Docker container for WINDOWS"
@if "%2" == "" goto args_count_wrong
@if "%3" == "" goto args_count_ok 

:args_count_wrong
@ECHO "Please run from the command line and "
@ECHO " supply three arguments: a Drake release (drake-YYYYMMDD),"
@ECHO " and a relative path to a directory to mount as /notebooks."
@PAUSE
@exit /b 1

:args_count_ok
docker pull bgheneti/shapeshift:%1
docker run -it -p 8080:8080 -p 7000-7010:7000-7010 --rm -v "%cd%\%2":/notebooks bgheneti/shapeshift:%1 /bin/bash -c "cd /notebooks && jupyter notebook --ip 0.0.0.0 --port 8080 --allow-root --no-browser"
