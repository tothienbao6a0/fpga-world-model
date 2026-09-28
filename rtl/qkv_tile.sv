// Streaming INT8 matrix-vector tile for one QKV projection weight block.
// One shared activation and LANES signed weights are accepted each ready/valid
// cycle. Biases and accumulators are signed INT32; host supplies exactly INPUTS
// columns, then done pulses with LANES packed INT32 results.
module qkv_tile #(
    parameter integer LANES = 16,
    parameter integer INPUTS = 400
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire [LANES*32-1:0]     biases,
    input  wire                    input_valid,
    input  wire signed [7:0]       activation,
    input  wire [LANES*8-1:0]      weights,
    output wire                    input_ready,
    output reg                     done,
    output reg [LANES*32-1:0]      results
);
    reg running;
    integer input_index;
    integer lane;

    assign input_ready = running;

    always @(posedge clk) begin
        if (!rst_n) begin
            running <= 1'b0;
            input_index <= 0;
            done <= 1'b0;
            results <= 0;
        end else begin
            done <= 1'b0;
            if (start && !running) begin
                running <= 1'b1;
                input_index <= 0;
                results <= biases;
            end else if (running && input_valid) begin
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    results[lane*32 +: 32] <= $signed(results[lane*32 +: 32])
                        + ($signed(activation) * $signed(weights[lane*8 +: 8]));
                end
                if (input_index == INPUTS - 1) begin
                    running <= 1'b0;
                    done <= 1'b1;
                end else begin
                    input_index <= input_index + 1;
                end
            end
        end
    end
endmodule
