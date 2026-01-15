package demo;

import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.annotations.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.concurrent.Executors;

public class App {

    static void matMulGPU(float[] A, float[] B, float[] C, int N) {
        for (@Parallel int i = 0; i < N; i++) {
            for (@Parallel int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    static void matMulCPU(float[] A, float[] B, float[] C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    static void matMulVirtualThreads(float[] A, float[] B, float[] C, int N)
            throws InterruptedException {

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {

            for (int row = 0; row < N; row++) {
                final int r = row;
                executor.submit(() -> {
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < N; k++) {
                            sum += A[r * N + k] * B[k * N + j];
                        }
                        C[r * N + j] = sum;
                    }
                });
            }

            executor.shutdown();
            while (!executor.isTerminated()) {
                Thread.sleep(1);
            }
        }
    }

    public static void main(String[] args) throws Exception {

        int N = 1024;
        int size = N * N;

        float[] A = new float[size];
        float[] B = new float[size];
        float[] C = new float[size];

        for (int i = 0; i < size; i++) {
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        // ================= TornadoVM =================
        TaskGraph tg = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
                .task("t0", App::matMulGPU, A, B, C, N)
                .transferToHost(DataTransferMode.UNDER_DEMAND, C);

        TornadoExecutionPlan plan =
                new TornadoExecutionPlan(tg.snapshot());

        // Warm-up GPU
        for (int i = 0; i < 100; i++) {
            plan.execute();
        }

        long tg0 = System.nanoTime();
        plan.execute();
        long tg1 = System.nanoTime();

        System.out.println("GPU (TornadoVM): "
                + (tg1 - tg0) / 1_000_000.0 + " ms");

        // ================= Virtual Threads =================
        long tv0 = System.nanoTime();
        matMulVirtualThreads(A, B, C, N);
        long tv1 = System.nanoTime();

        System.out.println("CPU Virtual Threads: "
                + (tv1 - tv0) / 1_000_000.0 + " ms");

        // ================= CPU =================
        long tc0 = System.nanoTime();
        matMulCPU(A, B, C, N);
        long tc1 = System.nanoTime();

        System.out.println("CPU sequencial: "
                + (tc1 - tc0) / 1_000_000.0 + " ms");


    }
}
