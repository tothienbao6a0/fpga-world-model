// Broadcast each streamed QKV weight to CANDIDATES independent activations.
// Result order is candidate-major, then lane. Arithmetic matches qkv_tile.
module qkv_candidate_tile #(
    parameter integer LANES = 16,
    parameter integer INPUTS = 400,
    parameter integer CANDIDATES = 4
) (
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  start,
    input  wire [LANES*32-1:0]                   biases,
    input  wire                                  input_valid,
    input  wire [CANDIDATES*8-1:0]               activations,
    input  wire [LANES*8-1:0]                    weights,
    output wire                                  input_ready,
    output reg                                   done,
    output reg [CANDIDATES*LANES*32-1:0]         results
);
    reg running;
    integer input_index;
    integer candidate;
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
                for (candidate = 0; candidate < CANDIDATES; candidate = candidate + 1)
                    results[candidate*LANES*32 +: LANES*32] <= biases;
            end else if (running && input_valid) begin
                for (candidate = 0; candidate < CANDIDATES; candidate = candidate + 1) begin
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        results[(candidate*LANES+lane)*32 +: 32] <=
                            $signed(results[(candidate*LANES+lane)*32 +: 32])
                            + ($signed(activations[candidate*8 +: 8])
                               * $signed(weights[lane*8 +: 8]));
                    end
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
