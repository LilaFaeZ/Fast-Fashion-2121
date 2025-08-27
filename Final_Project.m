% Two interacting opinion dimensions:
% X: Likelihood of purchasing fast fashion (0-1)
% Y: Concern Variable against fast fashion (0-1)

% Parameters
n = 200;        % Number of agents
ns = 15000;      % Number of steps
mu_polarized = 0.0005; % Strong opinions are harder to change
polarization_threshold = 0.85; % Threshold to consider an opinion polarized
r = 0.05; % Radius of influence
mu = 0.01 + 0.02*rand(n,1); % Each agent has unique learning rate
beta = 0.35*rand(n,1); % Concerns regarding fast fashion
gamma = 0.001 + 0.01*rand(n,1);% Fashion consumption influence on concern
delta = 0.001 + 0.005*rand(n,1); % Indifference variable to prevent polarization
y_normal = 0.25; %Y Normal is the average typical concern an agent should have
x_normal = 0.50; % Acceptable rate of purchasing fast fashion
advertisement_influence = 0.05 + 0.2*rand(n,1); % Unique media influence

% Initialize X and Y
x0 = rand(n,1); 
y0 = rand(n,1);

% Initialize matrices
X = x0; 
Y = y0; 

% Wealth Class Variable
wealth_class = rand(n,1); % Each agent is assigned a wealth class  
social_bias = rand(n,1)*.25; % 

% Set up Advertisement Exposure
advertisement_exposure = zeros(n, ns);
advertisement_exposure(:, 1) = rand(n, 1);

% Track X-Y correlation
cross_corr = zeros(ns,1); 

% Main simulation loop
for i = 1:ns

    % Calculate adaptive radii
    r_X = r - 0.05*Y(:,i); % Higher Y, the narrower their radius for fast fashion opinions
    r_Y = r - 0.02*X(:,i); % Fashion consumption reduces Concern variable
    
    % Calculate neighbor radius for X and Y
    a_X = zeros(n,1);
    for j = 1:n
        neighbors = abs(X(:,i) - X(j,i)) <= r_X(j);
        a_X(j) = mean(X(neighbors,i));
    end

    a_Y = zeros(n,1);
    for j = 1:n
        neighbors = abs(Y(:,i) - Y(j,i)) <= r_Y(j);
        a_Y(j) = mean(Y(neighbors,i));
    end
    
    % Enforce polarized mu
    mu_X = mu;
    mu_X(X(:,i) >= polarization_threshold | X(:,i) <= (1 - polarization_threshold)) = mu_polarized;
    
    mu_Y = mu;
    mu_Y(Y(:,i) >= polarization_threshold | Y(:,i) >= 1-polarization_threshold) = mu_polarized;
    
    % Calculates the user's social media usage for each step 
    if i < ns
        advertisement_exposure(:, i+1) = 0.15 * advertisement_exposure(:, i)...
            + 0.05 * rand(n, 1); % % updates social media for each agent at each time step
    end

    % Wealth Class effect 
    social_bias(wealth_class < 0.3) = 0.1 * (0.3 - wealth_class(wealth_class<0.3));
    social_bias(wealth_class > 0.81) = -1.0 * (wealth_class(wealth_class>0.81)-0.05);

    % Combined Needs and Influence
    increase_x = advertisement_influence .* advertisement_exposure(:,i) ...
    + social_bias;
   
    % Update Fast Fashion Likelihood Opinion (X)
    X(:,i+1) = X(:,i) + mu.*(a_X - X(:,i)) ...
    - beta.*Y(:,i).*X(:,i) ...
    + increase_x;

    % Enforce bounds for X
    X(:,i+1) = max(0, min(1, X(:,i+1)));
    
    % Update Concern Variable  (Y)
    Y(:,i+1) = Y(:,i) + mu.*(a_Y - Y(:,i))...
        - gamma.*max(X(:,i)-x_normal,0).*Y(:,i)...
        + delta .* max(y_normal - Y(:,i), 0); 

    % Enforce bounds for Y
    Y(:,i+1) = max(0, min(1, Y(:,i+1)));
    
    % Track correlation
    cross_corr(i) = corr(X(:,i), Y(:,i));
    
    % Live plotting every few steps
    if mod(i,100)==0 %change here for faster plotting
        subplot(2,2,1)
        plot(X(:,1:i)','Color',[0.7 0.1 0.1 0.1])
        title('Purchasing Fast Fashion Likelihood (X)')
        ylabel("Agents' Opinions from 0 to 1")
        xlabel('Time Step')
        ylim([0 1])
        
        subplot(2,2,3)
        plot(Y(:,1:i)','Color',[0.1 0.5 0.1 0.1])
        title('Concern Against Fast Fashion (Y)')
        ylabel("Agents' Opinions from 0 to 1")
        xlabel('Time Step')
        ylim([0 1])

        subplot(2,2,[2,4])
        plot(X(:,i),Y(:,i),'.')
        title('2D graph of X and Y')
        xlabel("Purchasing Fast Fashion Likelihood")
        ylabel("Concern Variable")
        axis equal
        axis([0,1,0,1])
        drawnow
    end
end

% Final correlation plot
figure
plot(cross_corr,'k','LineWidth',2)
title('X-Y Correlation Over Time')
xlabel('Time Step'), ylabel('Likelihood of Purchasing Fast Fashion and Concern against Fast Fashion')
ylim([-1 1])
grid on