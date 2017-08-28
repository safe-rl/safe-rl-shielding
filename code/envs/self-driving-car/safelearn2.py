def evaluate(operand, a, b):
    if operand:
        return a
    else:
        return b

def inv_evaluate(operand, a, b):
    if operand:
        return not a
    else:
        return not b 


class Shield(object):
    def __init__(self):
        self.s0 = False;
    def move(self,i1, i2, i3, i4, o1, o2, o3):
        tmp4 = evaluate(o1, True, False);
        tmp6 = inv_evaluate(o1, True, False);
        tmp5 = inv_evaluate(i4, True, tmp6);
        tmp3 = evaluate(i3, tmp4, tmp5);
        tmp8 = evaluate(i4, True, tmp4);
        tmp7 = evaluate(i3, tmp8, tmp4);
        tmp2 = evaluate(i2, tmp3, tmp7);
        tmp10 = evaluate(i3, True, False);
        tmp12 = evaluate(i4, tmp4, False);
        tmp11 = evaluate(i3, tmp12, tmp8);
        tmp9 = evaluate(i2, tmp10, tmp11);
        tmp1 = evaluate(i1, tmp2, tmp9);
        o1__1 = tmp1;

        tmp16 = evaluate(o2, True, False);
        tmp18 = inv_evaluate(o1, True, False);
        tmp17 = inv_evaluate(i4, True, tmp18);
        tmp15 = evaluate(i3, tmp16, tmp17);
        tmp20 = evaluate(i4, tmp16, tmp18);
        tmp22 = evaluate(o1, True, False);
        tmp21 = evaluate(i4, tmp22, tmp16);
        tmp19 = evaluate(i3, tmp20, tmp21);
        tmp14 = evaluate(i2, tmp15, tmp19);
        tmp25 = evaluate(i4, True, False);
        tmp26 = evaluate(i4, True, tmp16);
        tmp24 = evaluate(i3, tmp25, tmp26);
        tmp29 = inv_evaluate(o2, True, False);
        tmp28 = evaluate(i4, tmp22, tmp29);
        tmp27 = inv_evaluate(i3, tmp28, tmp29);
        tmp23 = evaluate(i2, tmp24, tmp27);
        tmp13 = evaluate(i1, tmp14, tmp23);
        o2__1 = tmp13;

        tmp33 = evaluate(o3, True, False);
        tmp35 = inv_evaluate(o1, tmp33, False);
        tmp34 = inv_evaluate(i4, True, tmp35);
        tmp32 = evaluate(i3, tmp33, tmp34);
        tmp40 = inv_evaluate(o3, True, False);
        tmp39 = evaluate(o2, True, tmp40);
        tmp38 = evaluate(o1, tmp39, True);
        tmp41 = evaluate(o1, True, tmp40);
        tmp37 = evaluate(i4, tmp38, tmp41);
        tmp43 = evaluate(o1, tmp33, False);
        tmp42 = inv_evaluate(i4, tmp43, tmp33);
        tmp36 = inv_evaluate(i3, tmp37, tmp42);
        tmp31 = evaluate(i2, tmp32, tmp36);
        tmp47 = evaluate(o1, True, tmp39);
        tmp46 = evaluate(i4, True, tmp47);
        tmp45 = evaluate(i3, True, tmp46);
        tmp49 = evaluate(i4, tmp41, tmp47);
        tmp50 = evaluate(i4, tmp38, tmp40);
        tmp48 = evaluate(i3, tmp49, tmp50);
        tmp44 = inv_evaluate(i2, tmp45, tmp48);
        tmp30 = evaluate(i1, tmp31, tmp44);
        o3__1 = tmp30;

        self.s0 = False;

        return (o1__1, o2__1, o3__1)