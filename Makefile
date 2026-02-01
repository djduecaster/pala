deploy:
	./deploy_jetson.sh

run:
	./run_jetson.sh

go:
	./deploy_jetson.sh && ./run_jetson.sh
