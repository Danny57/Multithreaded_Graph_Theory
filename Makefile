main.o: N-Length-Nodes.cu
	rm -rf N-Length-Nodes
	nvcc -g -o N-Length-Nodes N-Length-Nodes.cu
