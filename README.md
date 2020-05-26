flask-video-streaming
=====================

Supporting code for my article [video streaming with Flask](http://blog.miguelgrinberg.com/post/video-streaming-with-flask) and its follow-up [Flask Video Streaming Revisited](http://blog.miguelgrinberg.com/post/flask-video-streaming-revisited).


# scripts
```sh
sudo docker run -it --rm -p 5000:5000 stonewlg/flask-video-streaming:latest
sudo docker-compose up -d
sudo docker-compose logs -f
```

# action
```sh
http://dev-iot-k8s.chintcloud.net/api/plugins/rpc/oneway/a9646f90-9f16-11ea-87c9-b16ebce0e367
Header：X-Authorization：Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ1c2VydGVzdEBjaGludC5jb20iLCJzY29wZXMiOlsiVEVOQU5UX0FETUlOIl0sInVzZXJJZCI6IjZlZjA1YzQwLTEwYzEtMTFlYS05NDFmLWYxOGVmNmIxOWQ4YyIsImZpcnN0TmFtZSI6InRlc3QiLCJsYXN0TmFtZSI6InVzZXIiLCJlbmFibGVkIjp0cnVlLCJpc1B1YmxpYyI6ZmFsc2UsInRlbmFudElkIjoiNjAxNjA1ZDAtMTBjMS0xMWVhLTk0MWYtZjE4ZWY2YjE5ZDhjIiwicHJvamVjdElkIjoiMTM4MTQwMDAtMWRkMi0xMWIyLTgwODAtODA4MDgwODA4MDgwIiwiaXNzIjoiY2hpbnRjbG91ZC5pbyIsImlhdCI6MTU5MDQ3MzU1MywiZXhwIjoxNTkwNTU5OTUzfQ.K0RQy4F2qP7kbZO1wJ8r-BXCbUOHQ5MvfgqaXX-yhRjFcuvwLF2qfOtXraTuVHFgGDWBCxMXGCYTg4jv-nq5YQ
{
    "id": 1,
    "method": "startAIMonitor",
    "params": true
}
{
    "id": 1,
    "method": "stopAIMonitor",
    "params": true
}
```