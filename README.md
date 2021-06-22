# Keras REST API for image classification using convolutional neural networks with synaptic metaplasticity

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Synaptic metaplasticity is a biological phenomenon shortly defined as the plasticity
of synaptic plasticity, meaning that the previous history of the synaptic activity
determines its current plasticity. This phenomenon interferes with some of
the underlying mechanisms that are considered important in memory and learning
processes, such as long-term potentiation and long-term depression. In this
work, we provide an approach to include metaplasticity in convolutional neural
networks to enhance learning in image classification problems. This approach
consists of including metaplasticity as a weight update function in the backpropagation
stage of convolutional layers.

### Built With

* [Python](https://www.python.org/)
* [Flask RESTPlus](https://flask-restplus.readthedocs.io/en/stable/)
* [Swagger](https://swagger.io/)
* [Gunicorn](https://gunicorn.org/)
* [Docker](https://www.docker.com/)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Install and configure CUDA.
* Check your NVIDIA driver.
* Update the NVIDIA display driver.
* Check your PATH environment variable.



### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/vvives/metaplasticity-rest-api.git
   ```
2. Install python libraries with the following command:

    ```sh
    pip install -r requirements.txt
    ```
3. Execute the WSGI or the Dockerfile within the corresponding folder.

    ```sh
    python wsgi.py
    ```

<!-- USAGE EXAMPLES -->
## Usage

1. Go to the default URL http://localhost:5000/ and visualize the Swagger interface.
2. Select any image classification problem: MNIST, Fashion MNIST or CIFAR-10.
3. Select the convolutional neural network architecture endpoint: alexnet, googlenet or lenet.
4. Click the button Try it out.
5. Use the image by default or load any of your own.
6. Click the button Execute and wait for the predicted value.

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Víctor Vives - vvives@dtic.ua.es

Project Link: [https://github.com/vvives/metaplasticity-rest-api](https://github.com/vvives/metaplasticity-rest-api)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Bioinspired Engineering and Health Informatics research group](https://web.ua.es/en/ibis/ingenieria-bioinspirada-e-informatica-para-la-salud.html)
