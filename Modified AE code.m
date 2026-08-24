clear all; close all; clc;
rng(0);

num_samples = 100000;
L = 10;                                  
num_blocks = floor(num_samples / L);
SNR_dB = randi([0 20], num_blocks, 1);


x_real_bits = randi([0 1], num_samples, 1);
x_imag_bits = randi([0 1], num_samples, 1);
x = (2*x_real_bits - 1) + 1j*(2*x_imag_bits - 1);
x = x / sqrt(2);


h_block = (randn(num_blocks,1) + 1j*randn(num_blocks,1)) / sqrt(2);
h = repelem(h_block, L);

noise_std = repelem(10.^(-SNR_dB/20), L);
noise = noise_std .* (randn(num_samples,1) + 1j*randn(num_samples,1)) / sqrt(2);

y = h .* x + noise;

input_real = real(y);
input_imag = imag(y);

X_blocks_real = reshape( input_real(1:num_blocks*L), L, num_blocks).';
X_blocks_imag = reshape(input_imag(1:num_blocks*L), L, num_blocks).';
X_data = [X_blocks_real, X_blocks_imag];

train_size = round(0.8 * num_blocks);

X_train = X_data(1:train_size,:);
Y_train = X_train;

X_test = X_data(train_size+1:end,:);
Y_test = X_test;

layers = [

    featureInputLayer(2*L)

    %% ENCODER
    fullyConnectedLayer(128)
    reluLayer

    %% BOTTLENECK
    fullyConnectedLayer(2,'Name','bottleneck')

    %% DECODER
    fullyConnectedLayer(128)
    reluLayer

    fullyConnectedLayer(2*L)

    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',256, ...
    'InitialLearnRate',0.001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{X_test, Y_test}, ...
    'ValidationFrequency',300, ...
    'Plots','training-progress', ...
    'Verbose',false);

[auto_model, train_info] = trainNetwork( X_train, Y_train, layers, options);

predicted_latent = activations( auto_model, X_test, 'bottleneck', 'OutputAs','rows');
h_estimated = predicted_latent(:,1) + 1j*predicted_latent(:,2);

h_true_blocks = h_block(train_size+1:end);

mse_value = mean(abs(h_estimated - h_true_blocks).^2);
%plotting&smoothing
train_loss = train_info.TrainingLoss;
val_loss = train_info.ValidationLoss;

% remove NaN values
val_loss = val_loss(~isnan(val_loss));

x_train = 1:length(train_loss);

% validation is stored less frequently
val_x = linspace(1, length(train_loss), length(val_loss));

% interpolate validation to full resolution
val_interp = interp1(val_x, val_loss, x_train, 'linear', 'extrap');

smooth_train = movmean(train_loss, 30);
smooth_val = movmean(val_interp, 30);

figure;
semilogy(smooth_train, 'b', 'LineWidth', 2);
hold on;
semilogy(smooth_val, 'r', 'LineWidth', 2);

xlabel('Iterations');
ylabel('Loss (MSE)');
title('Training vs Validation Loss');
legend('Training Loss','Validation Loss');
grid on;
%for true&predicted one
h_true_smooth = movmean(real(h_true_blocks), 5);
h_est_smooth  = movmean(real(h_estimated), 5);

figure;

plot(real(h_true_smooth(1:200,1)), 'b','LineWidth',1.5);
hold on;
plot(real(h_est_smooth(1:200,1)), 'r--','LineWidth',1.5);
xlabel('Sample Index');
ylabel('Real Channel Coefficient');
title('True Channel vs Autoencoder Estimated Channel');
legend('True Channel','Estimated Channel');
grid on;