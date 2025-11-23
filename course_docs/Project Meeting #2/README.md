NOTES FOR FUTURE SELF:
> Discussed bug with the students: If you don't specify in the federation strategy the minimal number of required clients for the communication of the models, the server assumes is 2 (default parameter) and starts with the least number of clients and ends when the first clients have ended.
> If running on the FEP university cluster - use the same queue and the same node, otherwise, pointing to 0.0.0.0:8080 or 127.0.0.1:8080 is going to point to different actual IPs and this will mean that they are not communicating (server is waiting for the clients and clients give an error of "no server to talk to")


How to run on HAR:
One terminal for the server:
python3 server_har.py

One terminal for:
How to run all the clients at the same time, in the same terminal:
python3 client_har.py --server 0.0.0.0:8080 --csv train_client1.csv & python3 client_har.py --server 0.0.0.0:8080 --csv train_client2.csv & python3 client_har.py --server 0.0.0.0:8080 --csv train_client3.csv

How to run on Medical:
One terminal for the server:
python3 server_medical.py --federation_rounds 20 --train_rounds 3 --num_clients 3

One terminal for each client:
python3 client_medical.py --server_address "0.0.0.0:8080" --client_id 3(ITS ID!!!)


