
% ============================================================================
% （4）AdamW 优化器
% ============================================================================
classdef AdamWOptimizer < handle
    properties
        params
        lr
        beta1
        beta2
        eps
        weight_decay
        m
        v
        t
        weight_params
    end
    
    methods
        function obj = AdamWOptimizer(params, varargin)
            % 解析输入参数
            p = inputParser;
            addParameter(p, 'lr', 0.001);
            addParameter(p, 'betas', [0.9, 0.999]);
            addParameter(p, 'eps', 1e-8);
            addParameter(p, 'weight_decay', 0.0);
            parse(p, varargin{:});
            
            obj.params = params;
            obj.lr = p.Results.lr;
            obj.beta1 = p.Results.betas(1);
            obj.beta2 = p.Results.betas(2);
            obj.eps = p.Results.eps;
            obj.weight_decay = p.Results.weight_decay;
            
            % 初始化动量
            obj.m = struct();
            obj.v = struct();
            param_names = fieldnames(params);
            for i = 1:length(param_names)
                name = param_names{i};
                obj.m.(name) = zeros(size(params.(name)));
                obj.v.(name) = zeros(size(params.(name)));
            end
            obj.t = 0;
            
            % 识别权重参数
            obj.weight_params = {};
            for i = 1:length(param_names)
                name = param_names{i};
                if contains(name, 'W_') || contains(name, 'W_e') || contains(name, 'W_pred')
                    obj.weight_params{end+1} = name;
                end
            end
        end
        
        function step(obj, grads, lr)
            obj.t = obj.t + 1;
            if nargin < 3 || isempty(lr)
                current_lr = obj.lr;
            else
                current_lr = lr;
            end
            
            param_names = fieldnames(obj.params);
            for i = 1:length(param_names)
                name = param_names{i};
                param = obj.params.(name);
                grad = grads.(name);
                
                % 分层学习率
                layer_lr = current_lr;
                if contains(name, 'W_pred') || contains(name, 'b_pred')
                    layer_lr = current_lr * 0.1;
                elseif contains(name, 'layer0')
                    layer_lr = current_lr * 1.2;
                end
                
                % 更新矩估计
                obj.m.(name) = obj.beta1 * obj.m.(name) + (1 - obj.beta1) * grad;
                obj.v.(name) = obj.beta2 * obj.v.(name) + (1 - obj.beta2) * (grad .^ 2);
                
                % 偏差修正
                m_hat = obj.m.(name) / (1 - obj.beta1^obj.t);
                v_hat = obj.v.(name) / (1 - obj.beta2^obj.t);
                
                % AdamW 更新
                if ismember(name, obj.weight_params)
                    update = layer_lr * (m_hat ./ (sqrt(v_hat) + obj.eps) + obj.weight_decay * param);
                else
                    update = layer_lr * (m_hat ./ (sqrt(v_hat) + obj.eps));
                end
                
                obj.params.(name) = param - update;
            end
        end
        
        function state = state_dict(obj)
            state.m = obj.m;
            state.v = obj.v;
            state.t = obj.t;
        end
        
        function load_state_dict(obj, state_dict)
            obj.m = state_dict.m;
            obj.v = state_dict.v;
            obj.t = state_dict.t;
        end
    end
end
