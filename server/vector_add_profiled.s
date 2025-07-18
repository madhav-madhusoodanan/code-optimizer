.section .data
    vec1:    .word 10, 20, 30, 40     // First vector (4 integers, 128 bits total)
    vec2:    .word 5, 15, 25, 35      // Second vector (4 integers, 128 bits total)
    result:  .space 16                // Space for result (4 integers)
    
    // -- CONFIGURABLE --
    loop_count: .quad 100000             // Number of times to run the core process
    
    // Cycle counter storage
    start_cycles: .quad 0
    end_cycles:   .quad 0
    total_cycles: .quad 0
    
    // Output strings
    input_header: .ascii "Input Vectors:\n"
    input_header_len = . - input_header
    
    vec1_header: .ascii "Vector 1: ["
    vec1_header_len = . - vec1_header
    
    vec2_header: .ascii "Vector 2: ["
    vec2_header_len = . - vec2_header
    
    result_header: .ascii "\nVector Addition Result (after all iterations):\nResult:   ["
    result_header_len = . - result_header
    
    iterations_header: .ascii "\nIterations: "
    iterations_header_len = . - iterations_header
    
    cycles_header: .ascii "\nTotal CPU Cycles: "
    cycles_header_len = . - cycles_header
    
    comma_space: .ascii ", "
    comma_space_len = . - comma_space
    
    close_bracket_newline: .ascii "]\n"
    close_bracket_newline_len = . - close_bracket_newline
    
    newline: .ascii "\n"
    newline_len = . - newline
    
    // Buffer for integer to string conversion
    num_buffer: .space 24

.section .text
.global _start

_start:
    // Load addresses of vectors and result
    adrp x0, vec1
    add x0, x0, :lo12:vec1
    adrp x1, vec2
    add x1, x1, :lo12:vec2
    adrp x2, result
    add x2, x2, :lo12:result
    
    // Save addresses to callee-saved registers
    mov x19, x0    // vec1 address
    mov x20, x1    // vec2 address
    mov x21, x2    // result address
    
    // Print input header
    mov x0, #1     // stdout
    adrp x1, input_header
    add x1, x1, :lo12:input_header
    mov x2, #input_header_len
    bl print_string
    
    // Print Vector 1
    mov x0, #1
    adrp x1, vec1_header
    add x1, x1, :lo12:vec1_header
    mov x2, #vec1_header_len
    bl print_string
    bl print_vector_from_x19
    
    // Print Vector 2
    mov x0, #1
    adrp x1, vec2_header
    add x1, x1, :lo12:vec2_header
    mov x2, #vec2_header_len
    bl print_string
    bl print_vector_from_x20
    
    // Load loop count
    adrp x9, loop_count
    add x9, x9, :lo12:loop_count
    ldr x9, [x9]
    
    // --- OPTIMIZATION: Hoist constant loads out of the loop ---
    // The vectors are loaded into NEON registers once, before the loop starts.
    // Each ld1 instruction loads an entire 128-bit vector (4 x 32-bit words)
    // into a 128-bit NEON register (v0 and v1).
    ld1 {v0.4s}, [x19]   // Load vec1 into v0 register
    ld1 {v1.4s}, [x20]   // Load vec2 into v1 register
    
    // Read start cycle counter (ARM64 equivalent to RDTSC)
    mrs x3, cntvct_el0
    adrp x4, start_cycles
    add x4, x4, :lo12:start_cycles
    str x3, [x4]
    
    // --- Loop for core process ---
    // The loop now only contains the arithmetic operation, making it much faster.
    mov x10, #0 // Initialize loop counter
computation_loop:
    // Perform NEON vector addition on values already in registers
    add v2.4s, v0.4s, v1.4s    // Add vectors
    
    add x10, x10, #1      // Increment loop counter
    cmp x10, x9           // Compare with loop_count
    b.lt computation_loop // Branch if less than
    
    // Read end cycle counter
    mrs x3, cntvct_el0
    adrp x4, end_cycles
    add x4, x4, :lo12:end_cycles
    str x3, [x4]

    // --- OPTIMIZATION: Store the final result once after the loop completes ---
    st1 {v2.4s}, [x21]   // Store result from v2 register to memory
    
    // Calculate cycle difference
    adrp x4, start_cycles
    add x4, x4, :lo12:start_cycles
    ldr x5, [x4]
    adrp x4, end_cycles
    add x4, x4, :lo12:end_cycles
    ldr x6, [x4]
    sub x7, x6, x5
    adrp x4, total_cycles
    add x4, x4, :lo12:total_cycles
    str x7, [x4]
    
    // Print result header
    mov x0, #1
    adrp x1, result_header
    add x1, x1, :lo12:result_header
    mov x2, #result_header_len
    bl print_string
    bl print_vector_from_x21
    
    // Print number of iterations
    mov x0, #1
    adrp x1, iterations_header
    add x1, x1, :lo12:iterations_header
    mov x2, #iterations_header_len
    bl print_string
    adrp x4, loop_count
    add x4, x4, :lo12:loop_count
    ldr x0, [x4]
    bl int64_to_string
    mov x0, #1
    mov x2, x1
    adrp x1, num_buffer
    add x1, x1, :lo12:num_buffer
    bl print_string
    
    // Print total CPU cycles
    mov x0, #1
    adrp x1, cycles_header
    add x1, x1, :lo12:cycles_header
    mov x2, #cycles_header_len
    bl print_string
    
    // Convert cycles to string and print
    adrp x4, total_cycles
    add x4, x4, :lo12:total_cycles
    ldr x0, [x4]
    bl int64_to_string
    mov x0, #1     // stdout
    mov x2, x1     // length from int64_to_string
    adrp x1, num_buffer
    add x1, x1, :lo12:num_buffer
    bl print_string
    
    // Print newline
    mov x0, #1
    adrp x1, newline
    add x1, x1, :lo12:newline
    mov x2, #newline_len
    bl print_string
    
    // Exit syscall
    mov x0, #0
    mov x8, #93
    svc #0

// Print string function (x0=fd, x1=string, x2=length)
print_string:
    mov x8, #64    // write syscall
    svc #0
    ret

// Print vector from address in x19
print_vector_from_x19:
    mov x22, x30   // Save link register
    mov x23, x19   // Use x19 as source address
    bl print_vector_common
    mov x30, x22   // Restore link register
    ret

// Print vector from address in x20
print_vector_from_x20:
    mov x22, x30
    mov x23, x20   // Use x20 as source address
    bl print_vector_common
    mov x30, x22
    ret

// Print vector from address in x21
print_vector_from_x21:
    mov x22, x30
    mov x23, x21   // Use x21 as source address
    bl print_vector_common
    mov x30, x22
    ret

// Common vector printing logic (expects vector address in x23)
print_vector_common:
    mov x24, x30   // Save link register
    mov x25, #0    // Counter for elements
    
print_loop:
    // Load current element (32-bit word)
    ldr w26, [x23, x25, lsl #2]
    
    // Convert integer to string and print
    mov x0, x26
    bl int_to_string
    mov x0, #1     // stdout
    mov x2, x1     // length from int_to_string
    adrp x1, num_buffer
    add x1, x1, :lo12:num_buffer
    bl print_string
    
    // Increment counter and check if we are done
    add x25, x25, #1
    cmp x25, #4
    beq print_close_bracket
    
    // Print comma and space if not the last element
    mov x0, #1
    adrp x1, comma_space
    add x1, x1, :lo12:comma_space
    mov x2, #comma_space_len
    bl print_string
    b print_loop
    
print_close_bracket:
    mov x0, #1
    adrp x1, close_bracket_newline
    add x1, x1, :lo12:close_bracket_newline
    mov x2, #close_bracket_newline_len
    bl print_string
    mov x30, x24   // Restore link register
    ret

// Convert 32-bit integer to string (x0=number, returns length in x1)
// Result is placed in num_buffer
int_to_string:
    adrp x2, num_buffer
    add x2, x2, :lo12:num_buffer
    mov x3, x2     // Pointer to the start of the buffer
    add x2, x2, #11 // Pointer to the end of the integer space in the buffer
    mov w4, #10    // Divisor
    mov x5, #0     // Sign flag
    
    // Handle negative numbers
    cmp w0, #0
    bge convert_loop
    mov x5, #1     // Set sign flag
    neg w0, w0     // Make number positive
    
convert_loop:
    udiv w6, w0, w4    // Quotient
    msub w7, w6, w4, w0 // Remainder
    add w7, w7, #'0'   // Convert to ASCII character
    strb w7, [x2, #-1]! // Store character and pre-decrement pointer
    mov w0, w6         // Move quotient to be the new number
    cbnz w0, convert_loop // Continue if number is not zero
    
    // Add negative sign if it was a negative number
    cbz x5, calc_length
    mov w7, #'-'
    strb w7, [x2, #-1]!
    
calc_length:
    sub x1, x3, x2     // Calculate length
    add x1, x1, #11
    mov x4, #0
copy_loop: // This copy is needed because we write the string from the end of the buffer
    ldrb w5, [x2, x4]
    strb w5, [x3, x4]
    add x4, x4, #1
    cmp x4, x1
    blt copy_loop
    ret

// Convert 64-bit integer to string (x0=number, returns length in x1)
int64_to_string:
    adrp x2, num_buffer
    add x2, x2, :lo12:num_buffer
    mov x3, x2     // Start of buffer
    add x2, x2, #23 // End of buffer
    mov x4, #10    // Divisor
    mov x5, #0     // Sign flag
    
    // Handle negative numbers
    cmp x0, #0
    bge convert_loop_64
    mov x5, #1     // Set sign flag
    neg x0, x0     // Make positive
    
convert_loop_64:
    udiv x6, x0, x4    // Quotient
    msub x7, x6, x4, x0 // Remainder
    add x7, x7, #'0'   // Convert to ASCII
    strb w7, [x2, #-1]! // Store and pre-decrement
    mov x0, x6
    cbnz x0, convert_loop_64
    
    // Add negative sign if needed
    cbz x5, calc_length_64
    mov w7, #'-'
    strb w7, [x2, #-1]!
    
calc_length_64:
    sub x1, x3, x2     // Length
    add x1, x1, #23
    mov x4, #0
copy_loop_64:
    ldrb w5, [x2, x4]
    strb w5, [x3, x4]
    add x4, x4, #1
    cmp x4, x1
    blt copy_loop_64
    ret
