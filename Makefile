build:
	cmake -S emulator -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$$(python3 -m pybind11 --cmakedir)
	cmake --build build -j
test:
	pytest -q
clean:
	rm -rf build
