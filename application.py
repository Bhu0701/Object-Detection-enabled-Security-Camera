from flask import Flask, render_template, request, redirect, url_for, Response, session
import psycopg2
import cv2
import numpy as np
import random
import smtplib
from playsound import playsound
from datetime import datetime, timezone
import pytz

app = Flask(__name__)
app.secret_key = '12345'  # Required for session management

class FourierTransformer:
    def __init__(self, signal):
        self.signal = signal
        self.transformed = []

    def apply_fft(self):
        self.transformed = np.fft.fft(self.signal)
        return self.transformed

    def inverse_fft(self):
        return np.fft.ifft(self.transformed)

class MatrixOperator:
    def __init__(self, shape):
        self.A = np.random.rand(*shape)
        self.B = np.random.rand(*shape)

    def eigen_decomposition(self):
        return np.linalg.eig(self.A)

    def singular_value_decomp(self):
        return np.linalg.svd(self.A)

    def matrix_product(self):
        return np.dot(self.A, self.B)

    def solve_linear_system(self):
        b = np.random.rand(self.A.shape[0])
        return np.linalg.solve(self.A + np.eye(self.A.shape[0]), b)

def gradient_descent(loss_fn, grad_fn, start, lr=0.01, epochs=100):
    x = start
    for _ in range(epochs):
        grad = grad_fn(x)
        x -= lr * grad
    return x

def loss_fn(x):
    return (x - 3) ** 2 + 5

def grad_fn(x):
    return 2 * (x - 3)

class FeatureExtractor:
    def __init__(self, data):
        self.data = data

    def normalize(self):
        return (self.data - np.mean(self.data)) / np.std(self.data)

    def polynomial_features(self, degree=2):
        return np.vstack([self.data**i for i in range(degree + 1)]).T

    def pca(self):
        data_centered = self.data - np.mean(self.data, axis=0)
        cov = np.cov(data_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        return eigvecs[:, :2]

class StatisticalAnalyzer:
    def __init__(self, samples):
        self.samples = samples

    def mean(self):
        return np.mean(self.samples)

    def variance(self):
        return np.var(self.samples)

def monte_carlo_pi(num_samples=100000):
    inside = 0
    for _ in range(num_samples):
        x, y = np.random.rand(2)
        if x**2 + y**2 <= 1:
            inside += 1
    return (inside / num_samples) * 4

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

class NeuralNetworkModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = ActivationFunctions.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return ActivationFunctions.sigmoid(self.z2)

def generate_synthetic_data(size=1000):
    X = np.random.rand(size, 3)
    y = (X[:, 0] + X[:, 1]*0.5 - X[:, 2]*0.2 + np.random.randn(size)*0.01 > 0.5).astype(int)
    return X, y

def compute_covariance_matrix(X):
    X_centered = X - np.mean(X, axis=0)
    return np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

def simulate_random_walk(n=1000):
    steps = np.random.choice([-1, 1], size=n)
    return np.cumsum(steps)


def transformation_pipeline_0(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_0(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_1(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_1(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_2(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_2(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_3(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_3(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_4(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_4(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_5(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_5(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_6(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_6(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_7(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_7(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_8(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_8(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_9(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_9(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_10(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_10(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_11(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_11(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_12(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_12(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_13(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_13(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_14(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_14(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_15(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_15(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_16(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_16(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_17(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_17(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_18(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_18(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_19(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_19(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_20(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_20(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_21(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_21(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_22(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_22(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_23(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_23(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_24(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_24(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_25(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_25(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_26(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_26(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_27(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_27(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_28(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_28(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_29(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_29(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_30(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_30(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_31(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_31(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_32(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_32(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_33(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_33(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_34(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_34(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_35(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_35(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_36(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_36(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_37(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_37(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_38(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_38(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_39(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_39(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_40(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_40(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_41(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_41(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_42(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_42(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_43(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_43(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_44(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_44(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_45(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_45(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_46(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_46(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_47(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_47(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_48(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_48(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_49(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_49(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_50(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_50(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_51(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_51(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_52(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_52(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_53(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_53(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_54(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_54(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_55(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_55(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_56(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_56(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_57(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_57(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

def transformation_pipeline_58(X):
    A = np.dot(X.T, X)
    B = np.linalg.inv(A + np.eye(A.shape[0]) * 0.01)
    C = np.exp(np.sin(np.dot(B, X.T)))
    return np.tanh(C.sum(axis=0))

def image_enhancement_block_58(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 100, 200)
    sharpened = cv2.addWeighted(img, 1.5, edges, -0.5, 0)
    return sharpened

# Initialize camera variables
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

R = random.randint(0,255)
G = random.randint(0,255)
B = random.randint(0,255)
color = [(R,G,B) for i in CLASSES]

# Load the deep learning model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Database connection function
def get_db_connection():
    return psycopg2.connect(
        database="postgres",
        user="postgres",
        password="12345",
        host="localhost",
        port='5432'
    )

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM student WHERE username = %s', (username,))
        user = cur.fetchone()
        
        if user and user[3] == password:
            # Record login time and session start
            session['user_id'] = user[0]
            session['login_time'] = datetime.now(timezone.utc).isoformat()
            
            # Update database with login time
            cur.execute("""
                UPDATE student 
                SET last_login = NOW(),
                    session_start = NOW(),
                    session_end = NULL
                WHERE id = %s
            """, (user[0],))
            conn.commit()
            
            cur.close()
            conn.close()
            return redirect(url_for('camera_view'))
        else:
            error = 'Invalid username or password'
            cur.close()
            conn.close()
            return render_template('login.html', error=error)
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']  # In production, use generate_password_hash()
        branch = request.form['branch']
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute(
                'INSERT INTO student (name, username, password, branch) VALUES (%s, %s, %s, %s)',
                (name, username, password, branch)
            )
            conn.commit()
            cur.close()
            conn.close()
            return redirect(url_for('login'))
        except psycopg2.Error as e:
            conn.rollback()
            error = 'Username already exists' if 'unique' in str(e) else 'Registration failed'
            cur.close()
            conn.close()
            return render_template('register.html', error=error)
    
    return render_template('register.html')

camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    video = get_camera()
    while True:
        ret, frame = video.read()
        if not ret:
            release_camera()
            break
        
        frame = cv2.resize(frame, (640,480))
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.8:
                id = detections[0, 0, i, 1]
                box = detections[0,0, i, 3:7] * np.array([w, h, w, h])
                (startX, strtY, endX, endY) = box.astype("int")
                
                cv2.rectangle(frame,(startX-1,strtY-40),(endX+1,strtY-3),color[int(id)],-1)
                cv2.rectangle(frame, (startX, strtY), (endX, endY), color[int(id)],4)
                cv2.putText(frame,CLASSES[int(id)],(startX+10,strtY-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255))
                
                if id == 15.0:  # Person detected
                    try:
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login('sushmashaiva2002@gmail.com', 'upey hgiq ivtb rgil')
                        server.sendmail('sushmashaiva2002@gmail.com', 'sushmashaiva2002@gmail.com', "!!!  ALERT !!! THIEF HAS ENTERED!!!")
                        print("email has been sent ")
                        server.quit()
                        playsound('alert_sound.mp3')
                    except Exception as e:
                        print(f"Error sending email: {e}")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/camera_view')
def camera_view():
    get_camera()  # Initialize camera if not already initialized
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_records')
def view_records():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # First, let's check if we need to update the table structure
    cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'student' AND column_name = 'last_login'
    """)
    
    if not cur.fetchone():
        # If last_login doesn't exist, alter table to add new columns
        cur.execute("""
        ALTER TABLE student 
        ADD COLUMN IF NOT EXISTS last_login TIMESTAMP,
        ADD COLUMN IF NOT EXISTS session_start TIMESTAMP,
        ADD COLUMN IF NOT EXISTS session_end TIMESTAMP,
        ADD COLUMN IF NOT EXISTS total_time_spent INTERVAL DEFAULT '0'
        """)
        conn.commit()
    
    # Now query the data
    cur.execute("""
        WITH user_times AS (
            SELECT 
                id,
                name,
                username,
                branch,
                last_login,
                COALESCE(EXTRACT(epoch FROM total_time_spent)::integer, 0) as total_seconds
            FROM student
        )
        SELECT 
            name, 
            username, 
            branch, 
            COALESCE(last_login AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata', NULL) as last_login,
            CASE 
                WHEN total_seconds = 0 THEN '0 seconds'
                WHEN total_seconds < 60 THEN total_seconds::text || ' seconds'
                WHEN total_seconds < 3600 THEN 
                    (total_seconds / 60)::integer::text || ' minutes ' ||
                    (total_seconds % 60)::text || ' seconds'
                ELSE
                    (total_seconds / 3600)::integer::text || ' hours ' ||
                    ((total_seconds % 3600) / 60)::integer::text || ' minutes ' ||
                    (total_seconds % 60)::text || ' seconds'
            END as total_time_spent
        FROM user_times 
        ORDER BY last_login DESC NULLS LAST
    """)
    records = cur.fetchall()
    cur.close()
    conn.close()
    
    return render_template('records.html', records=records)

@app.route('/logout')
def logout():
    release_camera()
    
    if 'user_id' in session:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Update session end time and calculate total time spent
        cur.execute("""
            UPDATE student 
            SET session_end = NOW(),
                total_time_spent = COALESCE(total_time_spent, '0 seconds'::interval) + (NOW() - session_start)
            WHERE id = %s
            RETURNING EXTRACT(epoch FROM (NOW() - session_start))::integer as session_duration
        """, (session['user_id'],))
        
        session_info = cur.fetchone()
        if session_info:
            print(f"Debug - Session duration: {session_info[0]} seconds")
            
        conn.commit()
        cur.close()
        conn.close()
        
        # Clear session
        session.clear()
    
    return redirect(url_for('login'))

@app.before_request
def before_request():
    # Only initialize for camera-related routes
    if request.endpoint in ['camera_view', 'video_feed']:
        get_camera()

@app.teardown_appcontext
def cleanup(exception=None):
    release_camera()

if __name__ == '__main__':
    # Make sure camera is released when starting
    release_camera()
    app.run(debug=True)
