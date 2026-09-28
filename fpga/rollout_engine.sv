// Streaming latent rollout engine. Q8.8 state and weights, Q16 cost.
// A host streams HORIZON actions per candidate after start. The engine keeps
// weights and intermediate states local and returns the cheapest first action.
module rollout_engine #(
    parameter integer HORIZON = 4,
    parameter integer NUM_CANDIDATES = 9
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [15:0] initial_position,
    input  wire signed [15:0] initial_velocity,
    input  wire signed [15:0] target_position,
    input  wire        [63:0] position_weights,
    input  wire        [63:0] velocity_weights,
    input  wire               action_valid,
    input  wire signed  [1:0] action,
    output wire               action_ready,
    output reg                done,
    output reg signed   [1:0] best_action,
    output reg         [63:0] best_cost
);
    reg running;
    integer step_index;
    integer candidate_index;
    reg signed [15:0] state_position;
    reg signed [15:0] state_velocity;
    reg signed [15:0] saved_position;
    reg signed [15:0] saved_velocity;
    reg signed [15:0] saved_target;
    reg        [63:0] saved_position_weights;
    reg        [63:0] saved_velocity_weights;
    reg signed  [1:0] first_action;
    reg        [63:0] action_cost_acc;

    // Saturating dot product of four Q8.8 values. The arithmetic shift floors
    // negative products, matching the software oracle's bit-exact contract.
    function automatic signed [15:0] dot4;
        input [63:0] coeffs;
        input signed [15:0] position;
        input signed [15:0] velocity;
        input signed [15:0] action_q;
        reg signed [35:0] sum;
        reg signed [35:0] shifted;
        begin
            sum = $signed(coeffs[15:0]) * position
                + $signed(coeffs[31:16]) * velocity
                + $signed(coeffs[47:32]) * action_q
                + $signed(coeffs[63:48]) * 16'sd256;
            shifted = sum >>> 8;
            if (shifted > 36'sd32767)
                dot4 = 16'sd32767;
            else if (shifted < -36'sd32768)
                dot4 = -16'sd32768;
            else
                dot4 = shifted[15:0];
        end
    endfunction

    wire signed [15:0] action_q = $signed(action) * 16'sd256;
    wire signed [15:0] next_position = dot4(saved_position_weights, state_position, state_velocity, action_q);
    wire signed [15:0] next_velocity = dot4(saved_velocity_weights, state_position, state_velocity, action_q);
    wire        [63:0] action_penalty = action == 0 ? 64'd0 : 64'd2621;
    wire signed [16:0] position_error = $signed(next_position) - $signed(saved_target);
    wire        [33:0] position_square = position_error * position_error;
    wire        [31:0] velocity_square = $signed(next_velocity) * $signed(next_velocity);
    wire        [45:0] weighted_velocity = velocity_square * 14'd9830;
    wire        [47:0] terminal_cost = position_square + (weighted_velocity >> 16);
    wire        [63:0] candidate_cost = action_cost_acc + action_penalty + terminal_cost;
    wire signed  [1:0] candidate_first_action = step_index == 0 ? action : first_action;

    assign action_ready = running;

    always @(posedge clk) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            best_action <= 2'sd0;
            best_cost <= {64{1'b1}};
            step_index <= 0;
            candidate_index <= 0;
            state_position <= 0;
            state_velocity <= 0;
            saved_position <= 0;
            saved_velocity <= 0;
            saved_target <= 0;
            saved_position_weights <= 0;
            saved_velocity_weights <= 0;
            first_action <= 0;
            action_cost_acc <= 0;
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                running <= 1'b1;
                step_index <= 0;
                candidate_index <= 0;
                saved_position <= initial_position;
                saved_velocity <= initial_velocity;
                saved_target <= target_position;
                saved_position_weights <= position_weights;
                saved_velocity_weights <= velocity_weights;
                state_position <= initial_position;
                state_velocity <= initial_velocity;
                action_cost_acc <= 0;
                best_cost <= {64{1'b1}};
                best_action <= 0;
            end else if (running && action_valid) begin
                if (step_index == 0)
                    first_action <= action;
                if (step_index == HORIZON - 1) begin
                    if (candidate_cost < best_cost) begin
                        best_cost <= candidate_cost;
                        best_action <= candidate_first_action;
                    end
                    if (candidate_index == NUM_CANDIDATES - 1) begin
                        running <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        candidate_index <= candidate_index + 1;
                        step_index <= 0;
                        state_position <= saved_position;
                        state_velocity <= saved_velocity;
                        action_cost_acc <= 0;
                    end
                end else begin
                    step_index <= step_index + 1;
                    state_position <= next_position;
                    state_velocity <= next_velocity;
                    action_cost_acc <= action_cost_acc + action_penalty;
                end
            end
        end
    end
endmodule
